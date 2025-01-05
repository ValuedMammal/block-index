# `BlockIndex`

## Design goals
- We would like changes to the local chain to be monotone [issue #1005](https://github.com/bitcoindevkit/bdk/issues/1005), i.e. the aggregate changeset only grows, with no data removed
- We should be able to apply changes in any order and get the same result
- downside: large redundant data structure (block ids are represented twice, because every node must reference the block it connects to)

> `src/block_index.rs`

```rs
/// Maps height to a list of nodes
pub type BlockGraph<T> = BTreeMap<Height, Vec<Node<T>>>;

/// A condensed representation of the blockchain
pub struct BlockIndex<T> {
    // genesis node
    root: T,
    // block graph
    graph: BlockGraph<T>,
    // canonical index
    index: Vec<NodeId>,
    // chain tip iterator
    tip: List<T>,
}

/// A node in the block graph
pub struct Node<T> {
    // the inner type
    inner: T,
    // connected_to
    conn: BlockId,
}

// To be a node in the block graph, `T` must implement a trait

/// Defines the behavior of a node in the graph
pub trait BlockNode: Ord + Clone {
    /// Return the identity of this block
    fn block_id(&self) -> BlockId;
}
```

## Rationale
**Blocks can always be connected and in any order.** The only stipulation is that each newly connected block must specify the identity of the block it claims to build on, the so-called "connected-to" block.

```rs
/// Add `block` to the graph that connects to `conn`. 
pub fn connect(&mut self, block: T, conn: BlockId) { ... }
```

**The best chain is defined by a canonical index.** A valid chain consists of any node from which a continuous path may be traced back to genesis by a series of `connected_to` blocks. There may be more than one valid chain at a time. To find the current best chain we look for the highest non-contentious node in the graph (by height) that forms a valid chain. We refer to this node as the _tip_. The canonical chain is therefore defined by the path in the graph from the tip to the genesis block. We can represent the path succinctly as a `Vec<NodeId>`. If the next tip is contentious, we simply wait to declare a new tip until the longest chain emerges, relying instead on the last known chain tip.

We define another term, the _root_, as the first element in the canonical index, or equivalently, the last one added if walking backward from the tip. (This differs slightly from the genesis root node which is a special case). A graph root is typically located at the lowest height in the graph. To be a valid graph the root must either connect to genesis or connect to a block found in a parent graph. To illustrate, consider two examples: **1)** Block 1 is the root of a new parent graph. We know that block 1 must connect to the genesis block but the graph itself cannot contain block 0 because it would have nothing to connect to. **2)** A child graph consists of blocks 7, 8, and 9. In order to connect to the parent graph with chain tip 3, the root of the child graph (block 7) must reference block 3 by block id.

**Graphs can be combined** with the help of `merge_chains` which allows us to discover the blocks in another graph that do not already exist in the parent graph. This returns a changeset without actually modifying `self` and will error if the two chains don't connect. Assuming a changeset is returned, then applying the changes should be infallible.

```rs
/// Returns the set difference between two graphs.
pub fn merge_chains(&self, other: &BlockGraph<T>) -> Result<ChangeSet<T>, MergeChainsError> { ... }
```
**Changes are applied via a `ChangeSet<T>`** that consists of a vector of tuples representing the generic block and the id of the block it connects to.

```rs
pub struct ChangeSet<T> {
    // pairs of `(block, connected_to)`
    pub blocks: Vec<(T, BlockId)>,
}

/// Apply changeset to self.
fn apply_changeset(&mut self, changeset: &ChangeSet) { ... }
```

**We can iterate blocks in the chain.** The chain tip iterator is represented internally as a singly linked list. (here, `Node` differs from a node in the block graph referenced earlier)

> `src/list.rs`

```rs
/// A node in a linked list
struct Node<T> {
    value: T,
    next: Option<Arc<Node>>,
}

/// A linked list
struct List<T: Clone> {
    head: Option<Arc<Node<T>>>,
}

/// List iterator
pub struct ListIter<T> {
    current: Option<Arc<Node<T>>>,
}

impl<T: Clone> Iterator for ListIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> { ... }
}
```
