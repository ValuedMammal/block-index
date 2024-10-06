## `BlockIndex`

### Design goals/considerations
- We would like changes to the local chain to be monotone, i.e. the aggregate changeset only grows, with no data removed
- We should be able to apply changes in any order and get the same result

```rs
/// Maps height to a list of nodes
type BlockGraph = BTreeMap<Height, Vec<Node>>;

/// A local representation of the blockchain
struct BlockIndex {
    // genesis hash
    root: BlockHash,
    // block graph
    graph: BlockGraph,
    // canonical index
    index: Vec<NodeId>,
    // chain tip iterator
    tip: List<BlockId>,
}

struct Node {
    // hash of this block
    hash: BlockHash,
    // block connected-to
    conn: BlockId,
}

/// Identifies a node in the block graph. Represented as a tuple where
/// the first element is the block height and the second element
/// is an index into the list of nodes at that height. We use this
/// to quickly find nodes in the graph.
struct NodeId((Height, usize));
```

- downside: large redundant data structure (block ids are represented twice, because every node must reference the block it connects to)
- How are new blocks connected? Blocks can always be connected and in any order; the only stipulation is that each newly connected block must specify the block id it claims to build on, the so-called "connected_to".

```rs
/// Add `block` to the graph that connects to `conn`. 
pub fn connect(&mut self, block: BlockId, conn: BlockId) { ... }
```

- How to reconstruct the canonical chain? A valid chain consists of any node from which a continuous path may be traced back to genesis by a series of `connected_to` blocks. There may be more than one valid chain at a time. To find the current best chain we look for the highest non-contentious node in the graph (by height) that forms a valid chain. We refer to this node as the _tip_. The canonical chain is therefore defined by the path in the graph from the tip to the genesis block. We can represent the path succinctly as a `Vec<NodeId>`. If the next tip is contentious, we simply wait to declare a new tip until the longest chain emerges, relying instead on the last known chain tip.
- We define another term, the _root_, as the first element in the canonical path (or the last one added if walking backward from the tip). The root is typically located at the lowest height in the graph. To be a valid graph the root must either connect to genesis or connect to a block found in the parent graph. To illustrate, consider two examples: 1) Block 1 is the root of a new parent graph. We know that block 1 must connect to the genesis block but the graph itself cannot contain block 0 because by definition the genesis block does not connect to anything. 2) A child graph consists of blocks 7, 8, and 9. In order to connect to the parent graph with chain tip of block 3, the root of this graph (block 7) must reference block 3 by both height and hash.

```rs
/// Private method that iterates `self.graph` in reverse starting from the
/// leaves and follows every `connected_to` block back to genesis
fn reindex(&mut self) { ... }
```
- Using `merge_chains` allows us to discover the blocks in another graph that do not already exist in the parent graph. This returns a changeset without actually modifying `self` and will error if the two chains don't connect. Assuming a changeset is returned, then applying the changes should be infallible.
```rs
/// Returns the set difference between two graphs.
fn merge_chains(&self, other: Self) -> Result<ChangeSet, MergeChainsError> { ... }
```
- The `ChangeSet` consists of a vector of tuples representing a block and the one it connects to.

```rs
struct ChangeSet {
    // pairs of `(block, connected_to)`
    blocks: Vec<(BlockId, BlockId)>,
}

/// Apply changeset to self.
fn apply_changeset(&mut self, changeset: ChangeSet) { ... }
```

- Chain tip iterator

Similar to bdk's `CheckPoint` the chain tip iterator is a singly linked list. It's cheaply cloneable in the sense that nodes are atomically reference counted using `Arc`. One key difference is that the list iterator yields elements of `T` (typically a `BlockId`), rather than another list with the next head node.

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
```

### Issues
- stack overflow problem, see also https://github.com/bitcoindevkit/bdk/issues/1634
- should generalize Node to store extra info (e.g. header)
