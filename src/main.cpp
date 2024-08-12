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
    MoveInfo(): move_gid(-1), src_proc(-1), dst_proc(-1)   {}
    int move_gid;
    int src_proc;
    int dst_proc;
    bool block_info;                                                    // whether this message is about the moving block or about the links to it
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
    std::vector<double> x;                                              // some block data, e.g.
};

// callback for synchronous exchange, sending block move info and link info
void send_info(Block*                               b,                  // local block
               const diy::Master::ProxyWithLink&    cp,                 // communication proxy for neighbor blocks
               MoveInfo&                            sent_move_info,     // info to be sent
               int                                  dest_gid)           // gid of block on destination proc where to send the info
{
    // send block move info from src block to some other block on the destination proc
    sent_move_info.block_info = true;
    if (b->gid == sent_move_info.move_gid)
    {
        // enqueue potentially outside of the neighborhood by specifying gid and proc
        diy::BlockID dest_block = {dest_gid, sent_move_info.dst_proc};
        cp.enqueue(dest_block, sent_move_info);

        // debug
//         fmt::print(stderr, "send_info(): move info: gid {} enqueing move_gid {}, src_proc {} to dest_gid {} dst_proc {}\n",
//                 b->gid, sent_move_info.move_gid, sent_move_info.src_proc, dest_gid, sent_move_info.dst_proc);
    }

    // send link info from src block to all blocks linked to it
    sent_move_info.block_info = false;
    diy::Link*    l = cp.link();                                        // link to the neighbor blocks
    if (b->gid == sent_move_info.move_gid)
    {
        for (int i = 0; i < l->size(); ++i)
        {
            cp.enqueue(l->target(i), sent_move_info);

            // debug
//             fmt::print(stderr, "send_info(): link info: gid {} enqueing move_gid {}, src_proc {} dst_proc {} to gid {}\n",
//                 b->gid, sent_move_info.move_gid, sent_move_info.src_proc, sent_move_info.dst_proc, l->target(i).gid);
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
            recvd_move_info.src_proc    = recv_info.src_proc;
            recvd_move_info.dst_proc    = recv_info.dst_proc;
            recvd_move_info.block_info  = recv_info.block_info;

            // debug
//             fmt::print(stderr, "recv_info(): move_info: gid {} recvd_move_gid {} recvd_src_proc {}\n",
//                     b->gid, recvd_move_info.move_gid, recvd_move_info.src_proc);
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
                        link->neighbors()[j].proc = recv_info.dst_proc;
                }

                // debug
//                 fmt::print(stderr, "recv_info(): link_info: gid {} recvd_move_gid {} recvd_src_proc {} recvd_dst_proc {} from gid {}\n",
//                         b->gid, recv_info.move_gid, recv_info.src_proc, recv_info.dst_proc, l->target(i).gid);
            }
        }
    }
}

// set dynamic assigner blocks to local blocks of master
// TODO: make a version of DynamicAssigner ctor take master as an arg and do this
void SetDynamicAssigner(diy::DynamicAssigner&   dynamic_assigner,
                        diy::Master&            master)
{
    for (auto i = 0; i < master.size(); i++)
        dynamic_assigner.set_rank(master.communicator().rank(), master.gid(i));
}

// move one block from src to dst proc
// TODO: make this a member function of dynamic assigner
void MoveBlock(diy::DynamicAssigner&    assigner,
               diy::Master&             master,
               int&                     move_gid,                       // input/output, updated at the dst proc
               int&                     src_proc,                       // input/output, updated at the dst proc
               int&                     dst_proc)                       // input/output, updated at the dst proc
{
    // communicate info about the block moving from src to dst proc
    MoveInfo sent_move_info, recvd_move_info;                           // information about the block that is moving
    int dest_gid = -1;                                                  // gid of block where to send the move_info
    if (move_gid >= 0)
    {
        sent_move_info.move_gid     = move_gid;
        sent_move_info.src_proc     = src_proc;
        sent_move_info.dst_proc     = dst_proc;

        // get gid of a block on the dest proc
        // TODO: need a better, more efficient way, slow and not always reliable, sometimes fails to find the dest_gid
        for (auto i = 0; i < assigner.nblocks(); i++)
        {
            if (assigner.rank(i) == dst_proc)
            {
                dest_gid = i;
                break;
            }
        }
        if (dest_gid < 0 || dest_gid >= assigner.nblocks())
        {
            fmt::print(stderr, "MoveBlock() error: dest_gid {} invalid.\n", dest_gid);
            abort();
        }
    }
    else
    {
        sent_move_info.move_gid     = -1;
        sent_move_info.src_proc     = -1;
        sent_move_info.dst_proc     = -1;
    }

    // debug
    if (master.communicator().rank() == sent_move_info.src_proc)
    fmt::print(stderr, "proc {} moving block gid {} to proc {} dest_gid {}\n",
            sent_move_info.src_proc, sent_move_info.move_gid, sent_move_info.dst_proc, dest_gid);

    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { send_info(b, cp, sent_move_info, dest_gid); });
    master.exchange(true);                  // true: remote exchange
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { recv_info(b, cp, recvd_move_info, master); });

    // debug
    if (master.communicator().rank() == recvd_move_info.dst_proc)
    fmt::print(stderr, "proc {} getting block gid {} from proc {}\n",
            recvd_move_info.dst_proc, recvd_move_info.move_gid, recvd_move_info.src_proc);

    // move the block from src to dst proc
    void* send_b;
    Block* recv_b;
    if (master.communicator().rank() == sent_move_info.src_proc)
    {
        send_b = master.block(master.lid(sent_move_info.move_gid));
        diy::MemoryBuffer bb;
        master.saver()(send_b, bb);
        master.communicator().send(sent_move_info.dst_proc, 0, bb.buffer);
    }
    else if (master.communicator().rank() == recvd_move_info.dst_proc)
    {
        recv_b = static_cast<Block*>(master.creator()());
        diy::MemoryBuffer bb;
        master.communicator().recv(recvd_move_info.src_proc, 0, bb.buffer);
        recv_b->load(recv_b, bb);
    }

    // move the link for the moving block from src to dst proc and update master on src and dst proc
    if (master.communicator().rank() == sent_move_info.src_proc)
    {
        diy::Link* send_link = master.link(master.lid(sent_move_info.move_gid));
        diy::MemoryBuffer bb;
        diy::LinkFactory::save(bb, send_link);
        master.communicator().send(sent_move_info.dst_proc, 0, bb.buffer);

        // remove the block from the master
        Block* delete_block = static_cast<Block*>(master.get(master.lid(sent_move_info.move_gid)));
        master.release(master.lid(sent_move_info.move_gid));
        delete delete_block;
    }
    else if (master.communicator().rank() == recvd_move_info.dst_proc)
    {
        diy::MemoryBuffer bb;
        diy::Link* recv_link;
        master.communicator().recv(recvd_move_info.src_proc, 0, bb.buffer);
        recv_link = diy::LinkFactory::load(bb);

        // add block to the master
        master.add(recvd_move_info.move_gid, recv_b, recv_link);

        // update return arguments
        move_gid = recvd_move_info.move_gid;
        src_proc = recvd_move_info.src_proc;
        dst_proc = recvd_move_info.dst_proc;
    }
}

int main(int argc, char* argv[])
{
    diy::mpi::environment     env(argc, argv);                          // diy equivalent of MPI_Init
    diy::mpi::communicator    world;                                    // diy equivalent of MPI communicator
    int bpr = 8;                                                        // blocks per rank
    int                       nblocks = world.size() * bpr;             // total number of blocks in global domain

    diy::ContiguousAssigner   static_assigner(world.size(), nblocks);

    Bounds domain(3);                                                   // global data size
    domain.min[0] = domain.min[1] = domain.min[2] = 0;
    domain.max[0] = domain.max[1] = domain.max[2] = 255;

    diy::Master master(world,
                       1,                                               // one thread
                       -1,                                              // all blocks in memory
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
                         [&](int gid,                                   // block global id
                             const Bounds& core,                        // block bounds without any ghost added
                             const Bounds& bounds,                      // block bounds including ghost region
                             const Bounds& domain,                      // global data bounds
                             const RGLink& link)                        // neighborhood
                         {
                             Block*     b   = new Block;
                             RGLink*    l   = new RGLink(link);
                             b->gid         = gid;
                             b->bounds      = bounds;

                             master.add(gid, b, l);
                         });

    // debug: display the decomposition
//     master.foreach(&Block::show_block);

    // assume the source proc knows which block it wants to send and where
    // TODO: eventually a DIY algorithm will implement work stealing to determine the move information at the source proc
    // the move information will be communicated to other procs; they are not assumed to have global knowledge about the movement
    int move_gid = -1;                                                  // -1 indicates not a source of block movement
    int src_proc = -1;
    int dst_proc = -1;
    if (world.rank() == 0)
    {
        move_gid     = bpr - 1;
        src_proc     = 0;
        dst_proc     = 1;
    }

    // copy static assigner to dynamic assigner
    diy::DynamicAssigner    dynamic_assigner(world, world.size(), nblocks);
    SetDynamicAssigner(dynamic_assigner, master);                       // TODO: make a version of DynamicAssigner ctor take master and do this

    // move one block from src to dst proc
    MoveBlock(dynamic_assigner, master, move_gid, src_proc, dst_proc);  // TODO: make this a dynamic assigner member function

    // debug: print the master of src and dst proc
    if (world.rank() == src_proc)
    {
        fmt::print(stderr, "master size {}\n", master.size());
        for (auto i = 0; i < master.size(); i++)
            fmt::print(stderr, "lid {} gid {}\n", i, master.gid(i));
    }
    else if (world.rank() == dst_proc)
    {
        fmt::print(stderr, "master size {}\n", master.size());
        for (auto i = 0; i < master.size(); i++)
            fmt::print(stderr, "lid {} gid {}\n", i, master.gid(i));
    }
}
