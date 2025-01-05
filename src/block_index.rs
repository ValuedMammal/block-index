//! Block index

use alloc::vec::Vec;
use bitcoin::block::Header;
use bitcoin::hashes::Hash;
use core::fmt;
use serde::{Deserialize, Serialize};

use bdk_chain::collections::{BTreeMap, BTreeSet};
use bdk_chain::BlockId;
use bdk_chain::ChainOracle;
use bdk_chain::CheckPoint;
use bdk_chain::Merge;
use bitcoin::BlockHash;

use crate::ll;

/// Height
type Height = u32;

/// Block graph
pub type BlockGraph<T> = BTreeMap<Height, Vec<Node<T>>>;

/// Block header and height
pub type IndexedHeader = (Height, Header);

/// Defines the behavior of a node in the graph
pub trait BlockNode: Ord + Clone {
    /// block id
    fn block_id(&self) -> BlockId;
}

impl BlockNode for (Height, Header) {
    fn block_id(&self) -> BlockId {
        BlockId {
            height: self.0,
            hash: self.1.block_hash(),
        }
    }
}

impl BlockNode for BlockId {
    fn block_id(&self) -> BlockId {
        *self
    }
}

/// A node in the block graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node<T> {
    /// the inner type
    inner: T,
    /// connected to
    conn: BlockId,
}

impl<T: BlockNode> Node<T> {
    /// Construct a new node
    pub fn new(inner: T, conn: BlockId) -> Self {
        Self { inner, conn }
    }

    /// Connected to
    pub fn connected_to(&self) -> BlockId {
        self.conn
    }
}

impl<T> core::ops::Deref for Node<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Block index
#[derive(Debug, Clone, PartialEq)]
pub struct BlockIndex<T: BlockNode> {
    /// root
    pub root: T,
    /// graph
    pub graph: BlockGraph<T>,
    /// index
    pub index: Vec<NodeId>,
    /// chain tip
    pub tip: ll::List<T>,
}

impl<T: BlockNode + fmt::Debug> BlockIndex<T> {
    /// New
    pub fn new(root: T, graph: BlockGraph<T>) -> Self {
        // TODO: error if root block height != 0
        let mut idx = Self {
            root,
            graph,
            index: vec![],
            tip: ll::List::new(),
        };

        idx.reindex();

        idx
    }

    /// Create new from a list of blocks.
    ///
    /// Only use this when the given `blocks` appear in topological order, that is
    /// each subsequent block connects to the previous one.
    ///
    /// # Errors
    ///
    /// - If `blocks` does not contain the genesis block
    pub fn from_block_ids(
        blocks: impl IntoIterator<Item = T>,
    ) -> Result<Self, MissingGenesisError> {
        let mut graph = BlockGraph::new();
        let mut blocks = blocks.into_iter();
        let root = blocks.next().ok_or(MissingGenesisError::default())?;
        if root.block_id().height != 0 {
            return Err(MissingGenesisError::default());
        }
        let mut conn = root.clone();

        for block in blocks {
            let height = block.block_id().height;
            graph
                .entry(height)
                .or_default()
                .push(Node::new(block.clone(), conn.block_id()));
            conn = block;
        }

        Ok(Self::new(root, graph))
    }

    /// Get the graph nodes at a given height
    ///
    /// Note there are no nodes at height 0.
    fn nodes(&self, height: Height) -> Option<Vec<Node<T>>> {
        self.graph.get(&height).cloned()
    }

    /// Get the genesis node
    fn genesis_node(&self) -> Node<T> {
        Node {
            inner: self.root.clone(),
            conn: BlockId {
                height: 0,
                hash: BlockHash::all_zeros(),
            },
        }
    }

    /// Genesis block
    pub fn genesis_block(&self) -> BlockId {
        BlockId {
            height: 0,
            hash: self.genesis_hash(),
        }
    }

    /// Get the genesis hash
    pub fn genesis_hash(&self) -> BlockHash {
        self.root.block_id().hash
    }

    /// Construct the canonical chain from the current graph.
    ///
    /// ## The block graph
    ///
    /// Every graph has a root and a tip. The root is defined as the lowest block by height
    /// whose connected_to block does not appear in the graph. The tip is defined as the
    /// highest non-contentious block. To be a valid chain there must exist one unambiguous
    /// path from the tip to the root linked by a series of connected-to blocks. We also
    /// enforce in [`is_valid_chain`] that the root sits at the lowest part of the graph
    /// (height-wise).
    ///
    /// At each height we have a list of leaf nodes where each node contains a block and
    /// the one it connects to. The graph is monotone, so you can always add a node to the
    /// graph. To connect a block you must specify the block it connects to, which under
    /// normal circumstances will be the latest chain tip. Blocks may be connected in any
    /// order, and once connected, the node's location is permanent as referenced
    /// by its unique [`NodeId`].
    ///
    /// It's possible for two copies of the same block to point to different previous blocks
    /// (albeit at different heights) if the two previous blocks are also in the best chain.
    ///
    /// It's possible for two blocks contending for the same height to to connect to different
    /// blocks. This could happen in the case of a two-block reorg where the most recent
    /// common ancestor is two blocks behind the tip.
    ///
    /// This drawing shows the overall chain state at heights 0 to 4. The genesis block is
    /// implicit so it's not part of the graph. Every subsequent height contains a list of
    /// nodes where each node consists of a block hash and a `BlockId` of the block that
    /// this node claims to build on, the so-called "connected-to".
    ///
    /// 0: genesis
    /// 1: (H, B0)
    /// 2: (H, B1)
    /// 3: (H, B2), (Ha, B2), ..., (Hn, B2)
    pub fn reindex(&mut self) {
        // re-compute the canonical index from the current graph
        self.index = get_path(&self.graph).into_iter().collect();
        // reconstruct linked list
        self.create_tip();
        self._check_is_index_consistent();
    }

    /// Check that we have an index and that it's up to date with
    /// the chain tip
    fn _check_is_index_consistent(&self) {
        assert!(
            !self.index.is_empty() || self.iter().count() == 1,
            "index must not be empty"
        );

        // walk the nodes in the index and assert that they connect
        let mut conn = self.genesis_block();
        for id in &self.index {
            let node = self.graph.search(id).expect("block must exist in graph");
            assert_eq!(node.connected_to(), conn);
            let cur = BlockId {
                height: id.height(),
                hash: node.block_id().hash,
            };
            conn = cur;
        }

        let terminal_node = self.index.iter().last().copied().unwrap_or_default();
        let tip = self.search(&terminal_node).expect("block must exist");
        assert_eq!(
            self.get_chain_tip().unwrap(),
            tip.block_id(),
            "index out of sync with chain"
        );
    }

    /// Search for a node by id
    fn search(&self, id: &NodeId) -> Option<Node<T>> {
        if let NodeId::GENESIS = *id {
            return Some(self.genesis_node());
        }
        self.graph.search(id)
    }

    /// Construct chain tip iterator from the canonical index
    fn create_tip(&mut self) {
        self.tip = ll::List::new();

        // we push onto the list in ascending height order such that the chain
        // tip becomes the head of the list
        self.tip.push(self.root.clone());
        for id in &self.index {
            let node = self.search(id).expect("block must exist in graph");
            self.tip.push(node.inner);
        }
    }

    /// Iterate over blocks in the chain starting from the tip
    ///
    /// The returned [`ListIter`](ll::ListIter) is represented internally as a singly
    /// linked list.
    pub fn iter(&self) -> ll::ListIter<T> {
        self.tip.iter()
    }

    /// Return true if block is found in the graph regardless
    /// of whether it exists in the canonical chain
    pub fn scan(&self, block: BlockId) -> bool {
        if block == self.genesis_block() {
            return true;
        }
        let nodes = self.nodes(block.height).unwrap_or_default();
        for node in nodes {
            if node.block_id().hash == block.hash {
                return true;
            }
        }
        false
    }

    /// Initial changeset
    pub fn initial_changeset(&self) -> ChangeSet<T> {
        let mut blocks = vec![];

        // allow the genesis block to connect to itself
        let root = self.genesis_node();
        blocks.push((root.inner, root.conn));

        // collect graph nodes
        for nodes in self.graph.values() {
            for node in nodes {
                let conn = node.connected_to();
                blocks.push((node.inner.clone(), conn));
            }
        }

        ChangeSet { blocks }
    }

    /// Connect block to chain and return a [`ChangeSet`], which may be empty
    /// if the node represented by the given block already exists in `self`.
    ///
    /// Returns a tuple where the first element indicates whether the chain tip
    /// was successfully extended. This will be true if the given block connects
    /// to the current chain tip and false otherwise. Note that adding a block
    /// to the chain is infallible regardless of what it claims to connect to.
    pub fn connect(&mut self, block: T, conn: BlockId) -> (bool, ChangeSet<T>) {
        let mut changeset = ChangeSet::default();
        let mut extended = false;
        let height = block.block_id().height;

        let nodes = self.graph.entry(height).or_default();
        let nodes_len = nodes.len();
        let node = Node::new(block.clone(), conn);
        if nodes.contains(&node) {
            return (extended, changeset);
        }
        nodes.push(node);
        changeset.blocks.push((block.clone(), conn));
        let cur_tip = self.get_chain_tip().expect("must have chain tip");
        if conn == cur_tip {
            // we're extending the current tip with this block, so it's safe
            // to update the canonical index. note that the `index` of the new
            // node is equal to the length of the vec _before_ we pushed onto it.
            self.index.push(NodeId::new(height, nodes_len));
            self.tip.push(block);
            extended = true;
        }
        (extended, changeset)
    }

    /// Merge chains
    ///
    /// This will find the set difference between self and the given `graph` and return it
    /// as a [`ChangeSet`]. Errors if no point of connection exists.
    pub fn merge_chains(&self, other: &BlockGraph<T>) -> Result<ChangeSet<T>, MergeChainsError<T>> {
        let mut changeset = ChangeSet::default();

        // check is valid graph
        if !is_valid_chain(other) {
            return Err(MergeChainsError::InvalidChain);
        }

        // check point of connection
        let root_id = other.root();
        let node = other.search(&root_id).expect("root must exist");
        if !self.scan(node.connected_to()) {
            return Err(MergeChainsError::DoesNotConnect { root: node });
        }

        // for each node in the given graph, if the block does not exist in `self`
        // we add it to the changeset
        for (&height, nodes) in other {
            for node in nodes {
                let block = BlockId {
                    height,
                    hash: node.block_id().hash,
                };
                if !self.scan(block) {
                    changeset
                        .blocks
                        .push((node.inner.clone(), node.connected_to()));
                }
            }
        }

        Ok(changeset)
    }

    /// Apply changeset
    pub fn apply_changeset(&mut self, changeset: &ChangeSet<T>) {
        if changeset.blocks.is_empty() {
            return;
        }

        let mut chain_extended = true;

        // connect blocks
        for (block, conn) in &changeset.blocks {
            let (extended, _) = self.connect(block.clone(), *conn);
            if chain_extended && !extended {
                chain_extended = false;
            }
        }

        // we must reindex if any of the new blocks did not extend the
        // chain tip
        if !chain_extended {
            self.reindex();
        }
    }

    /// Convenience for calling merge_chains and then applying the resulting changeset
    pub fn apply_update(&mut self, graph: BlockGraph<T>) -> Result<(), MergeChainsError<T>> {
        let changeset = self.merge_chains(&graph)?;
        self.apply_changeset(&changeset);
        Ok(())
    }

    /// Get a block from the canonical chain if it exists at height.
    pub fn get(&self, height: Height) -> Option<BlockId> {
        self.iter()
            .map(|n| n.block_id())
            .find(|b| b.height == height)
    }

    /// Get the tip
    pub fn tip(&self) -> T {
        self.iter().next().expect("must have chain tip")
    }

    /// Iter checkpoints. Note: this is a temporary solution to provide interop with `bdk_wallet`
    pub fn iter_checkpoints(&self) -> CheckPoint {
        let mut blocks = BTreeSet::new();
        blocks.insert(self.genesis_block());
        blocks.extend(
            self.index
                .iter()
                .map(|id| self.get(id.height()).expect("must have block")),
        );
        CheckPoint::from_block_ids(blocks).expect("blocks must be in order")
    }

    /// Constructor from checkpoint. Note: this is a workaround that allows making
    /// a block graph from a checkpoint
    pub fn from_checkpoint(cp: CheckPoint) -> BlockIndex<BlockId> {
        let mut blocks = vec![];
        for cp in cp.iter() {
            let block = cp.block_id();
            let conn = cp.prev().unwrap_or(cp).block_id();
            blocks.push((block, conn));
        }
        blocks.reverse();
        let changeset = ChangeSet { blocks };
        BlockIndex::<BlockId>::from_changeset(changeset)
    }

    /// Get a reference to the block graph
    pub fn graph(&self) -> &BlockGraph<T> {
        &self.graph
    }

    /// Constructor from changeset. Panics if root not present in changeset.
    pub fn from_changeset(changeset: ChangeSet<T>) -> Self {
        let (root, _) = changeset
            .blocks
            .iter()
            .find(|(n, _)| n.block_id().height == 0)
            .cloned()
            .expect("changeset must include root");

        let changeset = ChangeSet {
            blocks: changeset
                .blocks
                .into_iter()
                .filter(|(n, _)| n.block_id().height > 0)
                .collect(),
        };

        let mut blk_idx = Self::new(root, BlockGraph::default());
        blk_idx.apply_changeset(&changeset);
        blk_idx
    }
}

/// Collect all [`NodeId`]s connecting the terminal node in the graph to the root.
///
/// The root is the last node in the graph whose connected_to block is not found
/// in this graph.
fn get_path<T: BlockNode>(graph: &BlockGraph<T>) -> BTreeSet<NodeId> {
    let mut path = BTreeSet::new();
    if graph.is_empty() {
        return path;
    }
    let mut height = *graph.keys().last().unwrap();
    // if the last height is contentious, i.e. more than one block candidates
    // exist at this height, we skip to the next depth in the graph until we find
    // an unambiguous tip
    while graph.get(&height).unwrap().len() > 1 {
        height -= 1;
    }
    let node = &graph.get(&height).unwrap()[0];
    let id = NodeId::new(height, 0);
    path.insert(id);
    let mut connected_to = node.connected_to();

    // We iterate the graph in reverse starting from the outermost nodes.
    // At each height, we walk the leaf nodes and if the current node is equal
    // to the current value of `connected_to`, we create an id for this node
    // and include it in the path index and finally update the `connected_to`
    // for the next level of the graph.
    for (&height, nodes) in graph.iter().rev() {
        for (index, node) in nodes.iter().enumerate() {
            if node.block_id() == connected_to {
                path.insert(NodeId::new(height, index));
                connected_to = node.connected_to();
                break;
            }
        }
    }

    path
}

/// Returns true if this block graph represents a valid chain. This is used to check
/// whether two chains can be merged cleanly
fn is_valid_chain<T: BlockNode>(graph: &BlockGraph<T>) -> bool {
    // must not be empty
    if graph.is_empty() {
        return false;
    }

    // computed path must contain the root, i.e. no gaps from tip to root
    let root_height = *graph.keys().next().unwrap();
    let path = get_path(graph);
    if path.iter().all(|n| n.height() > root_height) {
        return false;
    }

    true
}

/// The se of blocks to construct a block graph is missing the genesis block.
#[derive(Debug, Default)]
pub struct MissingGenesisError {}

impl fmt::Display for MissingGenesisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "missing genesis block")
    }
}

impl std::error::Error for MissingGenesisError {}

/// Error while trying to merge two chains
#[derive(Debug)]
pub enum MergeChainsError<T> {
    /// failed to connect to parent graph
    DoesNotConnect {
        /// The node that didn't connect
        root: Node<T>,
    },
    /// invalid chain
    InvalidChain,
}

impl<T: fmt::Debug> fmt::Display for MergeChainsError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DoesNotConnect { root } => {
                write!(f, "failed to apply changeset with root {root:?}")
            }
            Self::InvalidChain => write!(f, "invalid chain"),
        }
    }
}

impl<T: fmt::Debug + fmt::Display> std::error::Error for MergeChainsError<T> {}

/// Information for looking up a node in the block graph
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(pub(crate) (Height, usize));

impl NodeId {
    /// The genesis leaf reference
    const GENESIS: Self = Self((0, 0));

    /// New from height and index
    pub fn new(height: Height, index: usize) -> Self {
        Self((height, index))
    }

    /// Get the height of this node id
    pub fn height(&self) -> Height {
        self.0 .0
    }

    /// Get the index of this node id
    pub fn index(&self) -> usize {
        self.0 .1
    }
}

/// Change set
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChangeSet<T> {
    /// blocks
    pub blocks: Vec<(T, BlockId)>,
}

impl<T> ChangeSet<T> {
    /// New
    fn new() -> Self {
        Self { blocks: vec![] }
    }
}

impl<T> Default for ChangeSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Merge for ChangeSet<T> {
    fn merge(&mut self, other: Self) {
        self.blocks.extend(other.blocks);
    }

    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl<T: fmt::Debug + BlockNode> ChainOracle for BlockIndex<T> {
    type Error = core::convert::Infallible;

    fn get_chain_tip(&self) -> Result<BlockId, Self::Error> {
        Ok(self
            .iter()
            .next()
            .expect("chain must not be empty")
            .block_id())
    }

    fn is_block_in_chain(
        &self,
        block: BlockId,
        chain_tip: BlockId,
    ) -> Result<Option<bool>, Self::Error> {
        if block.height > chain_tip.height {
            return Ok(None);
        }

        // TODO: relax this by checking if `chain_tip`
        // is _reachable_ from self.tip
        if chain_tip != self.get_chain_tip().unwrap() {
            return Ok(None);
        }

        for cur in self.iter() {
            if cur.block_id() == block {
                return Ok(Some(true));
            }
        }

        Ok(Some(false))
    }
}

/// Trait
trait Tree<T> {
    /// The first element in the canonical index
    fn root(&self) -> NodeId;

    /// Search the tree for a block by id
    fn search(&self, id: &NodeId) -> Option<Node<T>>;
}

impl<T: BlockNode> Tree<T> for BlockGraph<T> {
    fn root(&self) -> NodeId {
        let index = get_path(self);
        index.iter().next().copied().unwrap_or_default()
    }

    fn search(&self, id: &NodeId) -> Option<Node<T>> {
        let nodes = self.get(&id.height())?;
        nodes.get(id.index()).cloned()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! block {
        ($height:expr, $hash:literal) => {
            BlockId {
                height: $height,
                hash: bitcoin::hashes::Hash::hash($hash.as_bytes()),
            }
        };
    }
    macro_rules! block_n {
        ($n:expr) => {
            BlockId {
                height: $n,
                hash: bitcoin::hashes::Hash::hash(($n as i32).to_be_bytes().as_slice()),
            }
        };
    }
    #[allow(unused_macros)]
    macro_rules! hash {
        ($h:literal) => {
            bitcoin::hashes::Hash::hash($h.as_bytes())
        };
    }
    macro_rules! node {
        ( $inner:expr, $conn:expr ) => {{
            Node {
                inner: $inner,
                conn: $conn,
            }
        }};
    }

    /// - `(0, hash!("G"))`
    /// - `(1, hash!("A"))`
    /// - `(2, hash!("B"))`
    /// - `(3, hash!("C"))`
    fn test_blocks() -> Vec<BlockId> {
        vec![
            block!(0, "G"),
            block!(1, "A"),
            block!(2, "B"),
            block!(3, "C"),
        ]
    }

    /// Returns a new `BlockIndex` with blocks from [`test_blocks`]
    fn test_chain_default() -> BlockIndex<BlockId> {
        let blocks = test_blocks();
        BlockIndex::from_block_ids(blocks).unwrap()
    }

    #[test]
    #[ignore = "expensive test"]
    fn reindex_massive_tree() {
        let mut graph = BlockGraph::new();
        let end = 10_000;
        let genesis = block!(0, "G");
        let mut conn = genesis;
        for h in 1u32..=end {
            let block = block_n!(h);
            graph.insert(h, vec![node!(block, conn)]);
            conn = block;
        }
        let chain = BlockIndex::new(genesis, graph);
        assert_eq!(chain.get_chain_tip().unwrap().height, end);
        assert_eq!(chain.iter().count(), (end + 1) as usize);
    }

    #[test]
    fn chain_oracle() {
        // test get chain tip
        let chain = test_chain_default();
        let tip = chain.get_chain_tip().unwrap();
        assert_eq!(tip, block!(3, "C"));

        // test is block in chain
        assert!(matches!(
            chain.is_block_in_chain(block!(4, "D"), tip),
            Ok(None),
        ));
        assert!(matches!(
            chain.is_block_in_chain(block!(3, "C"), tip),
            Ok(Some(true)),
        ));
        assert!(matches!(
            chain.is_block_in_chain(block!(2, "B"), tip),
            Ok(Some(true)),
        ));
        assert!(matches!(
            chain.is_block_in_chain(block!(1, "A"), tip),
            Ok(Some(true)),
        ));
        assert!(matches!(
            chain.is_block_in_chain(block!(0, "G"), tip),
            Ok(Some(true)),
        ));
        // test false
        assert!(matches!(
            chain.is_block_in_chain(block!(1, "A'"), tip),
            Ok(Some(false)),
        ));
    }

    #[test]
    fn iter_chain_tip() {
        let chain = test_chain_default();
        let mut it = chain.iter();
        let exp = block!(3, "C");
        assert_eq!(it.next(), Some(exp));
        // we started with 4 blocks and consumed 1
        assert_eq!(it.count(), 3);
    }

    #[test]
    fn contentious_tip() {
        let blocks = test_blocks();

        let block0 = blocks[0];
        let block1 = blocks[1];
        let block2 = blocks[2];
        let block3 = blocks[3];
        let block3_prime = block!(3, "C'");

        let mut tree = BTreeMap::<Height, Vec<Node<BlockId>>>::new();
        tree.insert(1, vec![node!(block1, block0)]);
        tree.insert(2, vec![node!(block2, block1)]);
        tree.insert(
            3,
            vec![
                node!(block3, block2),
                // contentious tip. chain should stop at last height - 1
                node!(block3_prime, block2),
            ],
        );
        let chain = BlockIndex::new(block!(0, "G"), tree);

        // expect
        let exp = vec![block2, block1, block0];
        assert_eq!(chain.iter().collect::<Vec<_>>(), exp);
    }

    #[test]
    fn sparse_chain() {
        let block0 = block!(0, "G");
        let block1 = block!(1, "A");
        let block2 = block!(2, "B");
        let block4 = block!(4, "C");
        let block8 = block!(8, "D");

        let tree: BTreeMap<Height, Vec<Node<BlockId>>> = [
            (1, vec![node!(block1, block0)]),
            (2, vec![node!(block2, block1)]),
            (4, vec![node!(block4, block2)]),
            (8, vec![node!(block8, block4)]),
        ]
        .into();
        let chain = BlockIndex::new(block!(0, "G"), tree);
        let it = chain.iter();
        assert_eq!(it.count(), 5);
    }

    #[test]
    fn test_reindex() {
        let blocks = test_blocks();
        let block0 = blocks[0];
        let block1 = blocks[1];
        let block2 = blocks[2];
        let block3 = blocks[3];

        let gen = block!(0, "G");
        // test new BlockIndex with an empty tree
        let chain = BlockIndex::new(gen, BlockGraph::new());
        assert!(
            chain.index.is_empty(),
            "index should not include genesis node id"
        );
        assert_eq!(chain.iter().count(), 1, "chain tip should include genesis");

        let mut graph = BTreeMap::new();
        graph.insert(1, vec![node!(block1, block0)]);
        graph.insert(2, vec![node!(block2, block1)]);
        graph.insert(3, vec![node!(block3, block2)]);

        let chain = BlockIndex::new(gen, graph);

        // calling `iter` should yield the expected blocks
        let mut exp = vec![];
        exp.push(block3);
        exp.push(block2);
        exp.push(block1);
        exp.push(block0);
        assert_eq!(chain.iter().collect::<Vec<_>>(), exp);
    }

    #[test]
    fn merge_trees() {
        // merging two trees should return the set difference
        let chain = test_chain_default(); // 0, 1, 2, 3

        let other_blocks = vec![
            block!(0, "G"),
            block!(1, "A'"),
            block!(2, "B'"),
            block!(3, "C'"),
        ];
        let other = BlockIndex::from_block_ids(other_blocks).unwrap();
        let cs = chain.merge_chains(&other.graph).unwrap();
        assert_eq!(cs.blocks.len(), 3);

        // merging same tree returns empty changeset
        let same = &chain.graph;
        assert_eq!(ChangeSet::default(), chain.merge_chains(same).unwrap());
    }

    #[test]
    fn test_apply_changeset() {
        let mut chain = test_chain_default(); // 0, 1, 2, 3

        let mut tree = BlockGraph::new();
        // connect two blocks. we should see chain index grow by two
        tree.insert(4, vec![node!(block!(4, "D"), block!(3, "C"))]);
        tree.insert(5, vec![node!(block!(5, "E"), block!(4, "D"))]);
        let changeset = chain.merge_chains(&tree).unwrap();

        chain.apply_changeset(&changeset);
        assert_eq!(chain.index.len(), 5);
        assert_eq!(chain.iter().count(), 6);
        assert_eq!(chain.iter().last().unwrap(), chain.genesis_block());
    }

    #[test]
    fn test_merge_chains_no_connect() {
        let chain = test_chain_default(); // 0, 1, 2, 3

        let mut tree = BlockGraph::new();
        let node5 = Node::new(block!(5, "E"), block!(4, "D"));
        tree.insert(5, vec![node5]);
        let node6 = Node::new(block!(6, "F"), block!(5, "E"));
        tree.insert(6, vec![node6]);
        let err = chain.merge_chains(&tree).unwrap_err();
        assert!(matches!(err, MergeChainsError::DoesNotConnect { root } if root == node5));
    }

    #[test]
    fn test_connect() {
        let mut chain = BlockIndex::new(block!(0, "G"), BlockGraph::new());
        let block1 = block!(1, "A");
        let block2 = block!(2, "B");
        let block3 = block!(3, "C");
        chain.connect(block1, block!(0, "G"));
        assert_eq!(chain.index.len(), 1);
        assert_eq!(chain.get_chain_tip().unwrap(), block1);

        chain.connect(block2, block1);
        assert_eq!(chain.index.len(), 2);
        assert_eq!(chain.get_chain_tip().unwrap(), block2);

        chain.connect(block3, block2);
        assert_eq!(chain.index.len(), 3);
        assert_eq!(chain.get_chain_tip().unwrap(), block3);

        // connecting a block that doesn't connect to chain tip should not extend
        // the index
        chain.connect(block!(3, "C'"), block2);
        assert_eq!(chain.index.len(), 3);
        assert_eq!(chain.get_chain_tip().unwrap(), block3);
    }

    #[test]
    fn test_reorg() {
        // test that chain can switch forks
        // 0-1-2-3
        //       3'-4-5

        let mut chain = test_chain_default();
        let block2 = block!(2, "B");

        // block 3 is a fork contender that should win because blocks 4, 5
        // build upon it
        let block3_prime = block!(3, "C'");
        let block4 = block!(4, "D");
        let block5 = block!(5, "E");
        let mut tree = BlockGraph::new();
        tree.insert(3, vec![node!(block3_prime, block2)]);
        tree.insert(4, vec![node!(block4, block3_prime)]);
        tree.insert(5, vec![node!(block5, block4)]);

        let changeset = chain.merge_chains(&tree).unwrap();
        assert!(is_valid_chain(&tree));
        chain.apply_changeset(&changeset);
        let id3 = chain.index[2];
        assert_eq!(chain.search(&id3).unwrap().block_id(), block3_prime);
    }

    #[test]
    fn test_is_valid_tree() {
        // tree must not be empty
        let mut tree = BlockGraph::new();
        assert!(!is_valid_chain(&tree));

        // path must span tree range
        tree = BlockGraph::new();
        let mut conn = block!(0, "G");
        for height in 1..=10 {
            let block = block_n!(height);
            tree.insert(height, vec![node!(block, conn)]);
            conn = block;
        }
        assert!(is_valid_chain(&tree));
        // introduce a gap by removing an entry, this tree is no longer
        // valid
        let _ = tree.remove(&5);
        assert!(!is_valid_chain(&tree));
    }

    #[test]
    fn test_tree_iter() {
        let blocks = test_blocks();
        let block0 = blocks[0];
        let block1 = blocks[1];
        let block2 = blocks[2];
        let block3 = blocks[3];

        let chain = test_chain_default();

        let mut it = chain.iter();
        assert_eq!(it.next(), Some(block3));
        assert_eq!(it.next(), Some(block2));
        assert_eq!(it.next(), Some(block1));
        assert_eq!(it.next(), Some(block0));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_from_blocks() {
        let blocks = vec![
            block!(0, "G"),
            block!(1, "A"),
            block!(2, "B"),
            block!(3, "C"),
        ];
        let chain = BlockIndex::from_block_ids(blocks).unwrap();
        assert_eq!(chain, test_chain_default());
        assert!(is_valid_chain(&chain.graph));
        assert_eq!(chain.get_chain_tip().unwrap(), block!(3, "C"));
    }

    #[test]
    fn test_from_changeset() {
        let chain = test_chain_default();
        let null_block = BlockId {
            height: 0,
            hash: BlockHash::all_zeros(),
        };
        let blocks = vec![
            (block!(0, "G"), null_block),
            (block!(1, "A"), block!(0, "G")),
            (block!(2, "B"), block!(1, "A")),
            (block!(3, "C"), block!(2, "B")),
        ];
        let changeset = ChangeSet { blocks };
        assert_eq!(chain.initial_changeset(), changeset);
        assert_eq!(BlockIndex::from_changeset(changeset), chain);
    }

    #[test]
    fn test_from_checkpoint() {
        let mut cp = CheckPoint::new(block!(0, "G"));
        cp = cp.push(block!(1, "A")).unwrap();
        cp = cp.push(block!(2, "B")).unwrap();
        cp = cp.push(block!(3, "C")).unwrap();

        // let default_chain = test_chain_default();
        // dbg!(default_chain.graph());
        assert_eq!(
            BlockIndex::<BlockId>::from_checkpoint(cp),
            test_chain_default()
        );
    }

    #[test]
    #[allow(unused)]
    #[ignore = "in develop"]
    fn header_test() {
        use bitcoin::hex::FromHex;

        let data_0 = <Vec<u8> as FromHex>::from_hex("0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4adae5494dffff001d1aa4ae18")
            .unwrap();
        let data_1 = <Vec<u8> as FromHex>::from_hex("0100000043497fd7f826957108f4a30fd9cec3aeba79972084e90ead01ea330900000000bac8b0fa927c0ac8234287e33c5f74d38d354820e24756ad709d7038fc5f31f020e7494dffff001d03e4b672")
            .unwrap();
        let data_2 = <Vec<u8> as FromHex>::from_hex("0100000006128e87be8b1b4dea47a7247d5528d2702c96826c7a648497e773b800000000e241352e3bec0a95a6217e10c3abb54adfa05abb12c126695595580fb92e222032e7494dffff001d00d23534")
            .unwrap();
        let header_0: Header = bitcoin::consensus::deserialize(&data_0).unwrap();
        let header_1: Header = bitcoin::consensus::deserialize(&data_1).unwrap();
        let header_2: Header = bitcoin::consensus::deserialize(&data_2).unwrap();

        let mut idx = BlockIndex::<IndexedHeader>::new((0, header_0), BlockGraph::new());
        // dbg!(&idx);

        idx.connect(
            (1, header_1),
            BlockId {
                height: 0,
                hash: header_1.prev_blockhash,
            },
        );
        idx.connect(
            (2, header_2),
            BlockId {
                height: 1,
                hash: header_2.prev_blockhash,
            },
        );

        dbg!(&idx);
    }
}
