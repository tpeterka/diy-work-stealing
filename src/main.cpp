#include <vector>
#include <iostream>
#include <bitset>

#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/master.hpp>

typedef     diy::DiscreteBounds         Bounds;
typedef     diy::RegularGridLink        RGLink;

// information about a block that is moving
struct MoveInfo
{
    MoveInfo(): move_gid(-1), src_rank(-1), dest_rank(-1)   {}
    int move_gid;
    int src_rank;
    int dest_rank;
};

// the block structure
struct Block
{
    Block() : bounds(0)                 {}
    static void*    create()            { return new Block; }
    static void     destroy(void* b)    { delete static_cast<Block*>(b); }

    static void save(const void* b_, diy::BinaryBuffer& bb)
    {
        const Block* b = static_cast<const Block*>(b_);

        diy::save(bb, b->gid);
        diy::save(bb, b->bounds);
        diy::save(bb, b->x);
    }

    static void load(void* b_, diy::BinaryBuffer& bb)
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

// callback for synchronous exchange, sending move info
void send_move_info(Block* b,                               // local block
                    const diy::Master::ProxyWithLink& cp,   // communication proxy for neighbor blocks
                    const MoveInfo& sent_move_info,         // info to be sent
                    int dest_gid)                           // gid of block on destination rank where to send the info
{
    // src block sends info to some other block in the destination rank
    if (b->gid == sent_move_info.move_gid)
    {
        // enqueue potentially outside of the neighborhood by specifying gid and proc
        diy::BlockID dest_block = {dest_gid, sent_move_info.dest_rank};
        cp.enqueue(dest_block, sent_move_info);
        fmt::print(stderr, "send_move_info(): enqueing move_gid {}, src_rank {} to dest_gid {} dest_rank {}\n",
                sent_move_info.move_gid, sent_move_info.src_rank, dest_gid, sent_move_info.dest_rank);
    }
}

// callback for synchronous exchange, receiving move info
void recv_move_info(Block* b,                               // local block
                    const diy::Master::ProxyWithLink& cp,   // communication proxy for neighbor blocks
                    MoveInfo& recvd_move_info)
{
    diy::Link*    l = cp.link();                // link to the neighbor blocks

    // dequeue incoming data, supposedly works even for remote source outside the neighborhood
    for (int i = 0; i < l->size(); ++i)
    {
        if (cp.incoming(i).size())
        {
            cp.dequeue(l->target(i).gid, recvd_move_info);
            fmt::print(stderr, "recv_move_info(): recvd_move_gid {} recvd_src_rank {}\n",
                    recvd_move_info.move_gid, recvd_move_info.src_rank);
        }
    }
}

// callback for synchronous exchange, sending link info
void send_link_info(Block* b,                               // local block
                    const diy::Master::ProxyWithLink& cp,   // communication proxy for neighbor blocks
                    const MoveInfo& sent_move_info)         // info to be sent
{
    diy::Link*    l = cp.link();                // link to the neighbor blocks

    MoveInfo temp_info;
    if (b->gid == sent_move_info.move_gid)
    {
        temp_info.move_gid  = sent_move_info.move_gid;
        temp_info.src_rank  = sent_move_info.src_rank;
        temp_info.dest_rank = sent_move_info.dest_rank;
    }
    else
    {
        temp_info.move_gid  = -1;
        temp_info.src_rank  = -1;
        temp_info.dest_rank = -1;
    }

    // src block sends info to all blocks linked to it
//     if (b->gid == sent_move_info.move_gid)
//     {
        for (int i = 0; i < l->size(); ++i)
        {
            cp.enqueue(l->target(i), temp_info);
            if (temp_info.move_gid != -1)
                fmt::print(stderr, "send_link_info(): gid {} enqueing move_gid {}, src_rank {} dst_rank {} to gid {}\n",
                        b->gid, temp_info.move_gid, temp_info.src_rank, temp_info.dest_rank, l->target(i).gid);
        }
//     }
}

// callback for synchronous exchange, receiving link info
void recv_link_info(Block* b,                               // local block
                    const diy::Master::ProxyWithLink& cp,   // communication proxy for neighbor blocks
                    diy::Master& master)
{
    diy::Link*  l = cp.link();                              // link to the neighbor blocks
    MoveInfo    recv_info;

    // dequeue incoming data
    if (b->gid != 3)
    {
        for (int i = 0; i < l->size(); ++i)
        {
            if (l->target(i).gid != b->gid && cp.incoming(i).size())
            {
                cp.dequeue(l->target(i).gid, recv_info);
                if (recv_info.move_gid != -1)
                {
                    fmt::print(stderr, "recv_link_info(): gid {} recvd_move_gid {} recvd_src_rank {} recvd_dst_rank {} from gid {}\n",
                            b->gid, recv_info.move_gid, recv_info.src_rank, recv_info.dest_rank, l->target(i).gid);
                    diy::Link* link = master.link(master.lid(b->gid));
                    for (auto j = 0; j < link->size(); j++)
                    {
                        if (link->neighbors()[j].gid == recv_info.move_gid)
                            link->neighbors()[j].proc = recv_info.dest_rank;
                    }
                }
            }
        }
    }
}

// callback for asynchronous iexchange
// return: true = I'm done unless more work arrives; false = I'm not done, please call me again
// bool move_info(Block* b,                       // local block
//          const diy::Master::ProxyWithLink& cp) // communication proxy for neighbor blocks
// {
//     diy::Link*    l = cp.link();               // link to the neighbor blocks
// 
//     MoveInfo info;
//     info.move_gid    = 3;
//     info.src_rank    = 0;
//     info.dest_rank   = 1;
//     MoveInfo recvd_info;
// 
//     do
//     {
//         // src block sends info to some other block in the destination rank
//         if (cp.gid() == info.move_gid)
//         {
//             // enqueue potentially outside of the neighborhood by specifying gid and proc
//             int dest_gid = bpr;                     // TODO: get the gid of a block on the dest_proc from the assigner
//             int dest_proc = 1;                      // MPI process of destination block
//             diy::BlockID dest_block = {dest_gid, dest_proc};
//             cp.enqueue(dest_block, info);
//             fmt::print(stderr, "move_info(): enqueing dest_gid {} dest_proc {}\n", dest_gid, dest_proc);
//         }
// 
//         // dequeue incoming data, works even for remote source outside the neighborhood
//         for (int i = 0; i < l->size(); ++i)
//         {
//             std::vector<int> incoming_gids;
// //             do
// //             {
//                 incoming_gids.clear();
//                 cp.incoming(incoming_gids);
//                 if (incoming_gids.size() && first_time)     // FIXME
//                 {
//                     fmt::print(stderr, "move_info(): incoming_gids [{}]\n", fmt::join(incoming_gids, ","));
//                     cp.dequeue(incoming_gids[0], recvd_info);
//                     fmt::print(stderr, "move_info(): recvd_move_gid {} recvd_src_rank {}\n",
//                             recvd_info.move_gid, recvd_info.src_rank);
//                     first_time = false;                     // FIXME
//                 }
// //             } while (incoming_gids.size());
//         }
//     } while (cp.fill_incoming());   // this loop is an optimization
// 
//     return true;
// }

int main(int argc, char* argv[])
{
    diy::mpi::environment     env(argc, argv);              // diy equivalent of MPI_Init
    diy::mpi::communicator    world;                        // diy equivalent of MPI communicator

    int bpr = 4;                // blocks per rank

    int                       nblocks = world.size() * bpr; // total number of blocks in global domain
    diy::ContiguousAssigner   static_assigner(world.size(), nblocks);

    Bounds domain(3);                                       // global data size
    domain.min[0] = domain.min[1] = domain.min[2] = 0;
    domain.max[0] = domain.max[1] = domain.max[2] = 255;

    diy::Master master(world,
                       1,                                   // one thread
                       -1,                                  // all blocks in memory
                       &Block::create,
                       &Block::destroy,
                       0,
                       &Block::save,
                       &Block::load);

    // create a regular decomposer and call its decompose function
    diy::RegularDecomposer<Bounds> decomposer(3,
                                              domain,
                                              nblocks);
    decomposer.decompose(world.rank(), static_assigner,
                         [&](int gid,                       // block global id
                             const Bounds& core,            // block bounds without any ghost added
                             const Bounds& bounds,          // block bounds including ghost region
                             const Bounds& domain,          // global data bounds
                             const RGLink& link)            // neighborhood
                         {
                             Block*     b   = new Block;
                             RGLink*    l   = new RGLink(link);
                             b->gid         = gid;
                             b->bounds      = bounds;

                             master.add(gid, b, l);    // add block to the master (mandatory)
                         });

    // debug: display the decomposition
//     master.foreach(&Block::show_block);

    // copy static assigner to dynamic assigner
    // each rank copies only its local blocks
    // TODO: add a diy copy constructor
    diy::DynamicAssigner    dynamic_assigner(world, world.size(), nblocks);
    for (auto i = 0; i < master.size(); i++)
        dynamic_assigner.set_rank(world.rank(), master.gid(i));

    // communicate move information BlockID (src_rank, gid) from src to dst

    // DEPRECATED, can't get p2p receive to probe src rank and receive from it
//     int move_gid, dest_rank;        // gid of block that is moving and the destination rank
//     if (world.rank() == 0)          // src knows what it wants to send to whom
//     {
//         move_gid = master.gid(master.size() - 1);
//         dest_rank = 1;
//         world.ssend(dest_rank, 0, move_gid);
//     }
//     else
//     {
//         diy::mpi::optional<diy::mpi::status> s;
//         s = world.iprobe(MPI_ANY_SOURCE, 0);
//         s = world.recv(MPI_ANY_SOURCE, 0, move_gid);
//         if (s)
//         {
//             fmt::print(stderr, "s = {}\n", s);
// //             world.recv(s->source(), 0, move_gid);
// //             world.recv(MPI_ANY_SOURCE, 0, move_gid);
//             fmt::print(stderr, "rank {} received move_gid {} from source rank {}\n", world.rank(), move_gid, s->source());
//         }
//     }

    MoveInfo sent_move_info, recvd_move_info;           // information about the block that is moving
    int dest_gid;                                       // gid of block where to send the move_info
    if (world.rank() == 0)                              // src rank knows about the move, TODO: decision logic
    {
        sent_move_info.move_gid     = bpr - 1;
        sent_move_info.src_rank     = 0;
        sent_move_info.dest_rank    = 1;
        dest_gid                    = bpr;              // TODO: get this from the assigner
    }
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { send_move_info(b, cp, sent_move_info, dest_gid); });
    master.exchange(true);                  // true: remote exchange
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { recv_move_info(b, cp, recvd_move_info); });

    // DEPRECATED, cannot figure out how to communicate remote data over iexchange
//     master.iexchange(&move_info);

    // move the block
    // TODO: eventually implement moving a block in a single diy function DynamicAssigner::move_block(gid, src_rank, dst_rank)

    // step 1: update assigner

    // debug: print dynamic assigner before the update
//     for (auto i = 0; i < nblocks; i++)
//         fmt::print(stderr, "Dynamic assigner before update: gid {} is on rank {}\n", i, dynamic_assigner.rank(i));

//     if (world.rank() == 0)
//     {
//         fmt::print(stderr, "rank 0 is moving gid {} to rank {}\n", sent_move_info.move_gid, sent_move_info.dest_rank);
//         dynamic_assigner.set_rank(sent_move_info.dest_rank, sent_move_info.move_gid);
//     }

    // debug: print dynamic assigner after the update
//     for (auto i = 0; i < nblocks; i++)
//         fmt::print(stderr, "Dynamic assigner after update: gid {} is on rank {}\n", i, dynamic_assigner.rank(i));

    // step 2: update link on all ranks that link to the moving block

    // debug: print link before the update
//     for (auto i = 0; i < master.size(); i++)
//     {
//         diy::Link* link = master.link(i);
//         fmt::print(stderr, "before update gid {} link ", master.gid(i));
//         for (auto j = 0; j < link->size(); j++)
//             fmt::print(stderr, "[gid {} proc {}] ", link->neighbors()[j].gid, link->neighbors()[j].proc);
//         fmt::print(stderr, "\n");
//     }

    // update the link
    // TODO: combine following exchange with previous one
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { send_link_info(b, cp, sent_move_info); });
    master.exchange();
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { recv_link_info(b, cp, master); });

    // debug: print link after the update
//     for (auto i = 0; i < master.size(); i++)
//     {
//         diy::Link* link = master.link(i);
//         fmt::print(stderr, "after update gid {} link ", master.gid(i));
//         for (auto j = 0; j < link->size(); j++)
//             fmt::print(stderr, "[gid {} proc {}] ", link->neighbors()[j].gid, link->neighbors()[j].proc);
//         fmt::print(stderr, "\n");
//     }

    // step 3: move the block from src to dst rank
    void* send_b;
    Block* recv_b;
    if (world.rank() == sent_move_info.src_rank)
    {
        send_b = master.block(master.lid(sent_move_info.move_gid));
        diy::MemoryBuffer bb;
        master.saver()(send_b, bb);
        world.send(sent_move_info.dest_rank, 0, bb.buffer);
    }
    else if (world.rank() == recvd_move_info.dest_rank)
    {
        recv_b = static_cast<Block*>(master.creator()());
        diy::MemoryBuffer bb;
        world.recv(recvd_move_info.src_rank, 0, bb.buffer);
        recv_b->load(recv_b, bb);
    }

    // step 4: move the link for the moving block from src to dst rank (TODO)
    if (world.rank() == sent_move_info.src_rank)
    {
        diy::Link* send_link = master.link(master.lid(sent_move_info.move_gid));
        diy::MemoryBuffer bb;
        diy::LinkFactory::save(bb, send_link);
        world.send(sent_move_info.dest_rank, 0, bb.buffer);

        // remove the block from the master
        // TODO: why doesn't master size decrease?
        master.release(master.lid(sent_move_info.move_gid));
        fmt::print(stderr, "master size {}\n", master.size());
    }
    else if (world.rank() == recvd_move_info.dest_rank)
    {
        diy::MemoryBuffer bb;
        diy::Link* recv_link;
        world.recv(recvd_move_info.src_rank, 0, bb.buffer);
        recv_link = diy::LinkFactory::load(bb);

        // add block to the master
        master.add(recvd_move_info.move_gid, recv_b, recv_link);
        fmt::print(stderr, "master size {}\n", master.size());
    }

    // debug: print link after the block move
//     for (auto i = 0; i < master.size(); i++)
//     {
//         diy::Link* link = master.link(i);
//         fmt::print(stderr, "after block move gid {} link ", master.gid(i));
//         for (auto j = 0; j < link->size(); j++)
//             fmt::print(stderr, "[gid {} proc {}] ", link->neighbors()[j].gid, link->neighbors()[j].proc);
//         fmt::print(stderr, "\n");
//     }
}
