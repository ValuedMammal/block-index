//! A cheaply cloneable singly-linked list

use std::sync::Arc;

#[derive(Debug, Clone)]
struct Node<T> {
    value: T,
    next: Option<Arc<Node<T>>>,
}

/// Linked list
#[derive(Debug, Clone, Default)]
pub struct List<T: Clone> {
    head: Option<Arc<Node<T>>>,
}

impl<T: Clone> List<T> {
    /// New
    pub fn new() -> Self {
        List { head: None }
    }

    /// Push
    pub fn push(&mut self, value: T) {
        let new_node = Arc::new(Node {
            value,
            next: self.head.take(),
        });
        self.head = Some(new_node);
    }

    /// Pop
    pub fn pop(&mut self) -> Option<T> {
        self.head.take().map(|node| {
            self.head = node.next.clone();
            node.value.clone()
        })
    }

    /// Return a new list iterator
    pub fn iter(&self) -> ListIter<T> {
        self.into_iter()
    }
}

impl<T: Clone + Eq> PartialEq for List<T> {
    fn eq(&self, other: &Self) -> bool {
        self.into_iter().eq(other)
    }
}

/// List iterator
pub struct ListIter<T> {
    current: Option<Arc<Node<T>>>,
}

impl<T: Clone> Iterator for ListIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.current.take().map(|node| {
            self.current = node.next.clone();
            node.value.clone()
        })
    }
}

impl<T: Clone> IntoIterator for &List<T> {
    type Item = T;
    type IntoIter = ListIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        ListIter {
            current: self.head.clone(),
        }
    }
}

impl<T: Clone> IntoIterator for List<T> {
    type Item = T;
    type IntoIter = ListIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        ListIter {
            current: self.head.clone(),
        }
    }
}

impl<T: Clone> Drop for List<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn iter() {
        let mut list = List::new();
        list.push(1);
        list.push(2);
        list.push(3);

        // test ListIter
        // cloning the list allows repeated iteration
        let mut it = list.clone().into_iter();
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next(), Some(1));
        assert!(it.next().is_none());

        // iterate again
        assert_eq!(list.into_iter().count(), 3);
    }

    macro_rules! block_n {
        ($n:expr) => {
            BlockId {
                height: $n,
                hash: bitcoin::hashes::Hash::hash(($n as i32).to_be_bytes().as_slice()),
            }
        };
    }

    #[test]
    #[ignore = "expensive test"]
    fn large_list() {
        use bdk_chain::BlockId;
        let mut list = List::<BlockId>::new();

        for n in 0..10_000 {
            list.push(block_n!(n));
        }

        dbg!(list.iter().count());
    }
}
