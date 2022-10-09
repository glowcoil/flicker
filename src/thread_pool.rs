#![allow(dead_code)]

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::{self, JoinHandle};

struct Context {
    // The thread pool can be in one of three states: quiescent, active, and exiting. The barrier
    // synchronizes transitions between these three states. A newly created thread pool starts in
    // the quiescent state. When `ThreadPool::run()` is called, the pool transitions to active and
    // threads begin executing tasks; when all tasks are finished, the pool transitions back to
    // quiescent and `ThreadPool::run()` returns. Finally, when the `ThreadPool` is dropped, the
    // pool transitions from quiescent to exiting, and all threads exit.
    barrier: Barrier,

    // During the quiescent state, `task` is only written to by the owner of the `ThreadPool`;
    // during the active and exiting states, `task` is read-only for all threads. Since state
    // transitions are synchronized by the barrier, this means that all accesses to `task` can be
    // `Relaxed`.
    task: AtomicPtr<*const (dyn Fn() + Sync)>,

    // During the active state, `panic` is used by up to one thread to publish a panic object;
    // during the quiescent state, the owner of the ThreadPool consumes that panic object if it was
    // published. Since the barrier acts to establish a happens-before relationship between the
    // acts of publishing and consuming the panic object, all accesses to `panic` can be `Relaxed`.
    panic: AtomicPtr<Box<dyn Any + Send>>,
}

impl Context {
    /// Attempt to store a panic object, but if one already exists then don't replace it
    fn store_panic(&self, panic: Box<dyn Any + Send>) {
        // Double-box the panic object since it's a wide pointer, which can't fit in an AtomicPtr
        let panic_ptr = Box::into_raw(Box::new(panic));

        // It's important that we use compare_exchange rather than compare_exchange_weak here,
        // since compare_exchange_weak can fail spuriously even if the comparison succeeds, which
        // would result in failing to propagate a panic.
        let result = self.panic.compare_exchange(
            ptr::null_mut(),
            panic_ptr,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );

        if result.is_err() {
            drop(unsafe { Box::from_raw(panic_ptr) });
        }
    }
}

/// A set of worker threads on which tasks can be executed in parallel.
pub struct ThreadPool {
    context: Arc<Context>,
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Constructs a thread pool with a number of threads equal to `num_threads`.
    ///
    /// Note that since [`ThreadPool::run`] executes tasks on the current thread as well as on
    /// worker threads, the effective number of workers will be `num_threads` + 1. In particular
    /// this means that when `num_threads` is 0, [`run`] will simply execute the given task on the
    /// current thread.
    ///
    /// [`run`]: ThreadPool::run
    pub fn with_threads(num_threads: usize) -> ThreadPool {
        let context = Arc::new(Context {
            barrier: Barrier::new(num_threads + 1),
            task: AtomicPtr::new(ptr::null_mut()),
            panic: AtomicPtr::new(ptr::null_mut()),
        });

        let mut handles = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let context = context.clone();

            let handle = thread::spawn(move || loop {
                // Transition from quiescent to either active or exiting state, depending on the
                // value of `context.task`
                context.barrier.wait();

                let task_ptr = context.task.load(Ordering::Relaxed);
                if task_ptr.is_null() {
                    break;
                }

                let task = unsafe { &**task_ptr };
                let result = panic::catch_unwind(AssertUnwindSafe(|| {
                    task();
                }));

                if let Err(panic) = result {
                    context.store_panic(panic);
                }

                // Transition from active to quiescent state
                context.barrier.wait();
            });

            handles.push(handle);
        }

        ThreadPool { context, handles }
    }

    /// Executes a task in the thread pool, blocking until all threads are finished.
    ///
    /// `task` will be invoked identically on every worker thread, as well as on the current
    /// thread. [`run`] will not return until all threads have finished executing the task.
    ///
    /// If a panic occurs on any thread during task execution, the call to [`run`] will also panic.
    /// If multiple panics occur, it is not deterministic which panic value will be propagated.
    ///
    /// [`run`]: ThreadPool::run
    pub fn run<F>(&mut self, task: F)
    where
        F: Fn() + Sync,
    {
        // A pointer to the task must be a wide pointer, which can't fit in an AtomicPtr, so store
        // a pointer to the wide pointer instead
        let task_wide_ptr = &task as &(dyn Fn() + Sync) as *const (dyn Fn() + Sync);
        let task_ptr =
            &task_wide_ptr as *const *const (dyn Fn() + Sync) as *mut *const (dyn Fn() + Sync);
        self.context.task.store(task_ptr, Ordering::Relaxed);

        // Transition from quiescent to active state
        self.context.barrier.wait();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            task();
        }));

        if let Err(panic) = result {
            self.context.store_panic(panic);
        }

        // Transition from active to quiescent state
        self.context.barrier.wait();

        self.context.task.store(ptr::null_mut(), Ordering::Relaxed);

        let panic = self.context.panic.swap(ptr::null_mut(), Ordering::Relaxed);
        if !panic.is_null() {
            panic::resume_unwind(*unsafe { Box::from_raw(panic) });
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Transition from quiescent to exiting state
        self.context.barrier.wait();

        for handle in self.handles.drain(..) {
            handle.join().unwrap();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn thread_pool() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        for num_threads in 0..4 {
            let mut pool = ThreadPool::with_threads(num_threads);

            let state = AtomicUsize::new(0);

            pool.run(|| {
                state.fetch_add(1, Ordering::Relaxed);
            });

            assert_eq!(state.load(Ordering::Relaxed), num_threads + 1);
        }
    }

    #[test]
    #[should_panic(expected = "panic in task")]
    fn panic_propagation() {
        let mut pool = ThreadPool::with_threads(8);

        pool.run(|| {
            panic!("panic in task");
        });
    }
}
