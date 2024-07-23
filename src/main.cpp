#include <vector>
#include <iostream>
#include <bitset>

#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/master.hpp>

typedef     diy::DiscreteBounds         Bounds;
typedef     diy::RegularGridLink        RGLink;

// the block structure
struct Block
{
    Block() : bounds(0)                 {}
    static void*    create()            { return new Block; }
    static void     destroy(void* b)    { delete static_cast<Block*>(b); }

    void save(const void* b_, diy::BinaryBuffer& bb)
    {
        const Block* b = static_cast<const Block*>(b_);

        diy::save(bb, b->gid);
        diy::save(bb, b->bounds);
        diy::save(bb, b->x);
    }

    void load(void* b_, diy::BinaryBuffer& bb)
    {
        Block* b = static_cast<Block*>(b_);

        diy::load(bb, b->gid);
        diy::load(bb, b->bounds);
        diy::load(bb, b->x);
    }

    void show_block(const diy::Master::ProxyWithLink& cp)
    {
        fmt::print(stderr, "Block {} bounds min [{}] max [{}]\n",
                cp.gid(), bounds.min, bounds.max);
    }

    // the block data
    int gid;
    Bounds bounds;
    std::vector<double> x;               // some block data, e.g.
};

int main(int argc, char* argv[])
{
    diy::mpi::environment     env(argc, argv);         // diy equivalent of MPI_Init
    diy::mpi::communicator    world;                   // diy equivalent of MPI communicator

    int                       nblocks = world.size() * 4;   // total number of blocks in global domain
    diy::ContiguousAssigner   static_assigner(world.size(), nblocks);

    Bounds domain(3);                                   // global data size
    domain.min[0] = domain.min[1] = domain.min[2] = 0;
    domain.max[0] = domain.max[1] = domain.max[2] = 255;

    diy::Master master(world,
                       1,                              // one thread
                       -1,                             // all blocks in memory
                       &Block::create,
                       &Block::destroy);

    // create a regular decomposer and call its decompose function
    diy::RegularDecomposer<Bounds> decomposer(3,
                                              domain,
                                              nblocks);
    decomposer.decompose(world.rank(), static_assigner,
                         [&](int gid,                           // block global id
                             const Bounds& core,                // block bounds without any ghost added
                             const Bounds& bounds,              // block bounds including ghost region
                             const Bounds& domain,              // global data bounds
                             const RGLink& link)                // neighborhood
                         {
                             Block*          b   = new Block;
                             RGLink*         l   = new RGLink(link);
                             b->bounds = bounds;

                             master.add(gid, b, l);    // add block to the master (mandatory)
                         });

    // debug: display the decomposition
    master.foreach(&Block::show_block);

    // copy static assigner to dynamic assigner
    // TODO: add a diy copy constructor, or is static -> dynamic even the right way to get to a dynamic assigner?
    diy::DynamicAssigner    dynamic_assigner(world, world.size(), nblocks);
    for (auto i = 0; i < nblocks; i++)
        dynamic_assigner.set_rank(static_assigner.rank(i), i);

    // move one block, eg. the last block of rank 0, to rank 1
    // we're not addressing how this decision was made; that's another problem
    // TODO: eventually implement moving a block in a single diy function DynamicAssigner::move_block(gid, src_rank, dst_rank)

    // step 1: update assigner

    // debug: print dynamic assigner before the update
    for (auto i = 0; i < nblocks; i++)
        fmt::print(stderr, "Dynamic assigner before update: gid {} is on rank {}\n", i, dynamic_assigner.rank(i));

    if (world.rank() == 0)
    {
        int move_gid = master.gid(master.size() - 1);
        fmt::print(stderr, "Rank 0 is moving gid {} to rank 1\n", move_gid);
        dynamic_assigner.set_rank(1, move_gid);
    }
    // debug: print dynamic assigner after the update
    for (auto i = 0; i < nblocks; i++)
        fmt::print(stderr, "Dynamic assigner after update: gid {} is on rank {}\n", i, dynamic_assigner.rank(i));

    // step 2: update link on all ranks that link to the moving block

    // debug: print link before the update
    for (auto i = 0; i < master.size(); i++)
    {
        diy::Link* link = master.link(i);
        fmt::print(stderr, "before update gid {} link ", master.gid(i));
        for (auto j = 0; j < link->size(); j++)
            fmt::print(stderr, "[gid {} proc {}] ", link->neighbors()[j].gid, link->neighbors()[j].proc);
        fmt::print(stderr, "\n");
    }

    // update the link
    for (auto i = 0; i < master.size(); i++)
    {
        diy::Link* link = master.link(i);
        link->fix(dynamic_assigner);
    }

    // debug: print link after the update
    for (auto i = 0; i < master.size(); i++)
    {
        diy::Link* link = master.link(i);
        fmt::print(stderr, "after update gid {} link ", master.gid(i));
        for (auto j = 0; j < link->size(); j++)
            fmt::print(stderr, "[gid {} proc {}] ", link->neighbors()[j].gid, link->neighbors()[j].proc);
        fmt::print(stderr, "\n");
    }

    // step 3: move the block from src to dst rank (TODO)
    Block* send_b;
    Block* recv_b;
    if (world.rank() == 0)              // assume src rank knows what to send where
    {
        send_b = static_cast<Block*>(master.block(master.size() - 1));
        diy::MemoryBuffer bb;
        send_b->save(send_b, bb);
        world.send(1, 0, bb);
    }
    else if (world.rank() == 1)         // TODO: rank 1 never learned that it was the dst or who is the src
    {
        recv_b = new Block;
        diy::MemoryBuffer bb;
        world.recv(0, 0, bb);
        recv_b->load(recv_b, bb);
    }

    // step 4: move the link for the moving block from src to dst rank (TODO)

    // step 5: update master for src and dst ranks (TODO)
    if (world.rank() == 0)
        master.release(master.size() - 1);
//     if (world.rank() == 1)
//         master.add(...)
}
