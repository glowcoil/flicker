#![allow(dead_code)]

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::{self, JoinHandle};

struct Context {
    barrier: Barrier,
    task: AtomicPtr<*const (dyn Fn() + Sync)>,
    panic: AtomicPtr<Box<dyn Any + Send>>,
}

impl Context {
    fn store_panic(&self, panic: Box<dyn Any + Send>) {
        let panic_ptr = Box::into_raw(Box::new(panic));

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

pub struct ThreadPool {
    context: Arc<Context>,
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
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

                context.barrier.wait();
            });

            handles.push(handle);
        }

        ThreadPool { context, handles }
    }

    pub fn run<F>(&mut self, task: F)
    where
        F: Fn() + Sync,
    {
        let task_wide_ptr = &task as &(dyn Fn() + Sync) as *const (dyn Fn() + Sync);
        let task_ptr =
            &task_wide_ptr as *const *const (dyn Fn() + Sync) as *mut *const (dyn Fn() + Sync);
        self.context.task.store(task_ptr, Ordering::Relaxed);

        self.context.barrier.wait();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            task();
        }));

        if let Err(panic) = result {
            self.context.store_panic(panic);
        }

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
