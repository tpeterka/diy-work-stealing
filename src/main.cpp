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
    bool block_info;     // whether this message is about the moving block or about the links to it
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

// callback for synchronous exchange, sending block move info and link info
void send_info(Block*                               b,              // local block
               const diy::Master::ProxyWithLink&    cp,             // communication proxy for neighbor blocks
               MoveInfo&                            sent_move_info, // info to be sent
               int                                  dest_gid)       // gid of block on destination rank where to send the info
{
    // send block move info from src block to some other block on the destination rank
    sent_move_info.block_info = true;
    if (b->gid == sent_move_info.move_gid)
    {
        // enqueue potentially outside of the neighborhood by specifying gid and proc
        diy::BlockID dest_block = {dest_gid, sent_move_info.dest_rank};
        cp.enqueue(dest_block, sent_move_info);

        // debug
//         fmt::print(stderr, "send_info(): move info: gid {} enqueing move_gid {}, src_rank {} to dest_gid {} dest_rank {}\n",
//                 b->gid, sent_move_info.move_gid, sent_move_info.src_rank, dest_gid, sent_move_info.dest_rank);
    }

    // send link info from src block to all blocks linked to it
    sent_move_info.block_info = false;
    diy::Link*    l = cp.link();                            // link to the neighbor blocks
    if (b->gid == sent_move_info.move_gid)
    {
        for (int i = 0; i < l->size(); ++i)
        {
            cp.enqueue(l->target(i), sent_move_info);

            // debug
//             fmt::print(stderr, "send_info(): link info: gid {} enqueing move_gid {}, src_rank {} dst_rank {} to gid {}\n",
//                 b->gid, sent_move_info.move_gid, sent_move_info.src_rank, sent_move_info.dest_rank, l->target(i).gid);
        }
    }
}

// callback for synchronous exchange, receiving move info and link info
void recv_info(Block*                               b,                  // local block
               const diy::Master::ProxyWithLink&    cp,                 // communication proxy for neighbor blocks
               MoveInfo&                            recvd_move_info,    // (output) received info
               diy::Master&                         master)
{
    diy::Link*  l = cp.link();                                          // link to the neighbor blocks
    MoveInfo    recv_info;

    // dequeue move and link info, including remote source outside the neighborhood
    std::vector<int> incoming_gids;
    cp.incoming(incoming_gids);
    for (size_t i = 0; i < incoming_gids.size(); i++)
    {
        int gid = incoming_gids[i];
        cp.dequeue(gid, recv_info);
        if (recv_info.block_info)                                       // this message was about the block being moved
        {
            recvd_move_info.move_gid    = recv_info.move_gid;
            recvd_move_info.src_rank    = recv_info.src_rank;
            recvd_move_info.dest_rank   = recv_info.dest_rank;
            recvd_move_info.block_info  = recv_info.block_info;

            // debug
//             fmt::print(stderr, "recv_info(): move_info: gid {} recvd_move_gid {} recvd_src_rank {}\n",
//                     b->gid, recvd_move_info.move_gid, recvd_move_info.src_rank);
        }
        else                                                            // this message was about the links to the moving block
        {
            if (cp.incoming(gid).size())
            {
                cp.dequeue(gid, recv_info);

                // update the link
                diy::Link* link = master.link(master.lid(b->gid));
                for (auto j = 0; j < link->size(); j++)
                {
                    if (link->neighbors()[j].gid == recv_info.move_gid)
                        link->neighbors()[j].proc = recv_info.dest_rank;
                }

                // debug
//                 fmt::print(stderr, "recv_info(): link_info: gid {} recvd_move_gid {} recvd_src_rank {} recvd_dst_rank {} from gid {}\n",
//                         b->gid, recv_info.move_gid, recv_info.src_rank, recv_info.dest_rank, l->target(i).gid);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    diy::mpi::environment     env(argc, argv);              // diy equivalent of MPI_Init
    diy::mpi::communicator    world;                        // diy equivalent of MPI communicator

    int bpr = 8;                // blocks per rank

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

                             master.add(gid, b, l);
                         });

    // debug: display the decomposition
//     master.foreach(&Block::show_block);

    // step 1: copy static assigner to dynamic assigner
    // each rank copies only its local blocks
    // TODO: add a diy copy constructor
    diy::DynamicAssigner    dynamic_assigner(world, world.size(), nblocks);
    for (auto i = 0; i < master.size(); i++)
        dynamic_assigner.set_rank(world.rank(), master.gid(i));

    // step 2: communicate info about the block moving from src to dst rank
    MoveInfo sent_move_info, recvd_move_info;           // information about the block that is moving
    int dest_gid;                                       // gid of block where to send the move_info
    if (world.rank() == 0)                              // src rank knows about the move, TODO: decision logic
    {
        sent_move_info.move_gid     = bpr - 1;
        sent_move_info.src_rank     = 0;
        sent_move_info.dest_rank    = 1;
        dest_gid                    = bpr;              // TODO: get this from the assigner

        // debug
        fmt::print(stderr, "rank {} moving block gid {} to rank {}\n",
                sent_move_info.src_rank, sent_move_info.move_gid, sent_move_info.dest_rank);
    }
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { send_info(b, cp, sent_move_info, dest_gid); });
    master.exchange(true);                  // true: remote exchange
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { recv_info(b, cp, recvd_move_info, master); });

    // debug
    if (world.rank() == recvd_move_info.dest_rank)
    fmt::print(stderr, "rank {} getting block gid {} from rank {}\n",
            recvd_move_info.dest_rank, recvd_move_info.move_gid, recvd_move_info.src_rank);

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

    // step 4: move the link for the moving block from src to dst rank and update master on src and dst rank
    if (world.rank() == sent_move_info.src_rank)
    {
        diy::Link* send_link = master.link(master.lid(sent_move_info.move_gid));
        diy::MemoryBuffer bb;
        diy::LinkFactory::save(bb, send_link);
        world.send(sent_move_info.dest_rank, 0, bb.buffer);

        // remove the block from the master
        Block* delete_block = static_cast<Block*>(master.get(master.lid(sent_move_info.move_gid)));
        master.release(master.lid(sent_move_info.move_gid));
        delete delete_block;

        // debug
        fmt::print(stderr, "master size {}\n", master.size());
        for (auto i = 0; i < master.size(); i++)
            fmt::print(stderr, "lid {} gid {}\n", i, master.gid(i));
    }
    else if (world.rank() == recvd_move_info.dest_rank)
    {
        diy::MemoryBuffer bb;
        diy::Link* recv_link;
        world.recv(recvd_move_info.src_rank, 0, bb.buffer);
        recv_link = diy::LinkFactory::load(bb);

        // add block to the master
        master.add(recvd_move_info.move_gid, recv_b, recv_link);

        // debug
        fmt::print(stderr, "master size {}\n", master.size());
        for (auto i = 0; i < master.size(); i++)
            fmt::print(stderr, "lid {} gid {}\n", i, master.gid(i));
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
