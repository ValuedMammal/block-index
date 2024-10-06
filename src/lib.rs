//! Block index

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

#[macro_use]
extern crate std;

mod block_index;
pub mod ll;

pub use block_index::*;
