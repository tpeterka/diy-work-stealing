#include <vector>
#include <iostream>
#include <bitset>

#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/master.hpp>

#include "opts.h"

#define WORK_MAX    100                 // maximum work a block can have (in some user-defined units)

typedef     diy::DiscreteBounds         Bounds;
typedef     diy::RegularGridLink        RGLink;

// information about a process' workload
struct WorkInfo
{
    int proc_rank;          // mpi rank of this process
    int top_gid;            // gid of most expensive block in this process TODO: can be top-k-gids, as long as k is fixed and known by all
    int top_work;           // work of top_gid TODO: can be vector of top-k work, as long as k is fixed and known by all
    int proc_work;          // total work of this process
    int nlids;              // local number of blocks in this process
};

// information about a block that is moving
struct MoveInfo
{
    MoveInfo(): move_gid(-1), src_proc(-1), dst_proc(-1)   {}
    MoveInfo(int move_gid_, int src_proc_, int dst_proc_) : move_gid(move_gid_), src_proc(src_proc_), dst_proc(dst_proc_) {}
    int move_gid;
    int src_proc;
    int dst_proc;
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
        fmt::print(stderr, "Block {} bounds min [{}] max [{}] work {}\n",
                gid, bounds.min, bounds.max, work);
    }

    void compute(const diy::Master::ProxyWithLink&  cp,
                 int                                max_time,               // maximum time for a block to compute
                 int                                iter)                   // curent iteration
    {
        useconds_t usec = max_time * work * 10000;

        // debug
//         fmt::print(stderr, "iteration {} block gid {} computing for {} s.\n", iter, gid, double(usec) / 1e6);

        usleep(usec);
    }

    // the block data
    int                 gid;
    Bounds              bounds;
    std::vector<double> x;                                              // some block data, e.g.
    int                 work;                                           // some estimate of how much work this block involves
};

// exchange work information among all processes
// TODO: make a master member function
void exchange_work_info(const diy::Master&      master,
                        std::vector<WorkInfo>&  all_work_info)          // (output) global work info
{
    auto nlids  = master.size();                    // my local number of blocks
    auto nprocs = master.communicator().size();     // global number of procs

    WorkInfo my_work_info = { master.communicator().rank(), -1, 0, 0, (int)nlids };

    // compile my work info
    for (auto i = 0; i < master.size(); i++)
    {
        Block* block = static_cast<Block*>(master.block(i));
        my_work_info.proc_work += block->work;
        if (my_work_info.top_gid == -1 || my_work_info.top_work < block->work)
        {
            my_work_info.top_gid    = block->gid;
            my_work_info.top_work   = block->work;
        }
    }

    // debug
//     fmt::print(stderr, "exchange_work_info(): proc_rank {} top_gid {} top_work {} proc_work {} nlids {}\n",
//             my_work_info.proc_rank, my_work_info.top_gid, my_work_info.top_work, my_work_info.proc_work, my_work_info.nlids);

    // vectors of integers from WorkInfo
    std::vector<int> my_work_info_vec =
    {
        my_work_info.proc_rank,
        my_work_info.top_gid,
        my_work_info.top_work,
        my_work_info.proc_work,
        my_work_info.nlids
    };
    std::vector<std::vector<int>>   all_work_info_vec;

    // exchange work info TODO: use something like distributed consensus protocol in iexchange?
    diy::mpi::all_gather(master.communicator(), my_work_info_vec, all_work_info_vec);

    // unpack received info into vector of structs
    all_work_info.resize(nprocs);
    for (auto i = 0; i < nprocs; i++)
    {
        all_work_info[i].proc_rank = all_work_info_vec[i][0];
        all_work_info[i].top_gid   = all_work_info_vec[i][1];
        all_work_info[i].top_work  = all_work_info_vec[i][2];
        all_work_info[i].proc_work = all_work_info_vec[i][3];
        all_work_info[i].nlids     = all_work_info_vec[i][4];
    }
}

// determine move info from work info
// the user writes this function
// returns true: move a block, false: don't move anything
bool decide_move_info(const diy::Master&            master,
                      const std::vector<WorkInfo>&  all_work_info,          // global work info
                      MoveInfo&                     move_info)              // (output) move info
{
    // parse all_work_info to decide block migration, all procs arriving at the same decision
    // for now pick the proc with the max. total work and move its top_gid to the proc with the min. total work
    // TODO later we can be more sophisticated, e.g., cut the work difference in half, etc., see related literature for guidance
    WorkInfo max_work = {-1, -1, 0, 0, 0};
    WorkInfo min_work = {-1, -1, 0, 0, 0};
    for (auto i = 0; i < master.communicator().size(); i++)             // for all process ranks
    {
        if (max_work.proc_rank == -1 || all_work_info[i].proc_work > max_work.proc_work)
        {
            max_work.proc_rank  = all_work_info[i].proc_rank;
            max_work.top_gid    = all_work_info[i].top_gid;
            max_work.top_work   = all_work_info[i].top_work;
            max_work.proc_work  = all_work_info[i].proc_work;
            max_work.nlids      = all_work_info[i].nlids;
        }
        if (min_work.proc_rank == -1 || all_work_info[i].proc_work < min_work.proc_work)
        {
            min_work.proc_rank  = all_work_info[i].proc_rank;
            min_work.top_gid    = all_work_info[i].top_gid;
            min_work.top_work   = all_work_info[i].top_work;
            min_work.proc_work  = all_work_info[i].proc_work;
            min_work.nlids      = all_work_info[i].nlids;
        }
    }

    // src and dst procs need to differ, and don't leave a proc with no blocks
    if (max_work.proc_rank != min_work.proc_rank && max_work.nlids > 1)
    {
        move_info.move_gid    = max_work.top_gid;
        move_info.src_proc    = max_work.proc_rank;
        move_info.dst_proc    = min_work.proc_rank;
        return true;
    }
    else
        return false;

    // debug
//     if (master.communicator().rank() == src_proc)
//         fmt::print(stderr, "exchange_work_info(): move_gid {} src_proc {} dst_proc {}\n",
//                 move_gid, src_proc, dst_proc);
}

// callback for synchronous exchange, sending link info
void send_link_info(Block*                              b,                  // local block
                    const diy::Master::ProxyWithLink&   cp,                 // communication proxy for neighbor blocks
                    const MoveInfo&                     move_info)          // info to be sent
{
    // send link info from src block to all blocks linked to it
    diy::Link*    l = cp.link();                                        // link to the neighbor blocks
    if (b->gid == move_info.move_gid)
    {
        for (auto i = 0; i < l->size(); ++i)
        {
            cp.enqueue(l->target(i), move_info);

            // debug
//             fmt::print(stderr, "send_link_info(): link info: gid {} enqueing move_gid {}, src_proc {} dst_proc {} to target({}) = gid {} proc {}\n",
//                 b->gid, move_info.move_gid, move_info.src_proc, move_info.dst_proc, i, l->target(i).gid, l->target(i).proc);
        }
    }
}

// callback for synchronous exchange, receiving link info
void recv_link_info(Block*                               b,                  // local block
                    const diy::Master::ProxyWithLink&    cp,                 // communication proxy for neighbor blocks
                    diy::Master&                         master)
{
    MoveInfo    move_info;

    diy::Link*    l = cp.link();
    for (int i = 0; i < l->size(); ++i)
    {
        int gid = l->target(i).gid;
        if (cp.incoming(gid).size())
        {
            cp.dequeue(gid, move_info);

            // update the link
            diy::Link* link = master.link(master.lid(b->gid));
            for (auto j = 0; j < link->size(); j++)
            {
                if (link->neighbors()[j].gid == move_info.move_gid)
                    link->neighbors()[j].proc = move_info.dst_proc;
            }

            // debug
//          fmt::print(stderr, "recv_info(): link_info: gid {} recvd_move_gid {} recvd_src_proc {} recvd_dst_proc {} from gid {}\n",
//                 b->gid, move_info.move_gid, move_info.src_proc, move_info.dst_proc, l->target(i).gid);
        }
    }
}

// set dynamic assigner blocks to local blocks of master
// TODO: make a version of DynamicAssigner ctor take master as an arg and do this
void set_dynamic_assigner(diy::DynamicAssigner&   dynamic_assigner,
                          diy::Master&            master)
{
    std::vector<std::tuple<int, int>> rank_gids(master.size());
    int rank = master.communicator().rank();

    for (auto i = 0; i < master.size(); i++)
        rank_gids[i] = std::make_tuple(rank, master.gid(i));

    dynamic_assigner.set_ranks(rank_gids);
}

// move one block from src to dst proc
// TODO: make this a member function of dynamic assigner
void  move_block(diy::DynamicAssigner&   assigner,
                 diy::Master&            master,
                 const MoveInfo&         move_info)
{
    // debug
//     if (master.communicator().rank() == src_proc)
//         fmt::print(stderr, "move_block(): move_gid {} src_proc {} dst_proc {}\n", move_gid, src_proc, dst_proc);

    // update links of blocks that neighbor the moving block
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { send_link_info(b, cp, move_info); });
    master.exchange();
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { recv_link_info(b, cp, master); });

    // move the block from src to dst proc
    void* send_b;
    Block* recv_b;
    if (master.communicator().rank() == move_info.src_proc)
    {
        send_b = master.block(master.lid(move_info.move_gid));
        diy::MemoryBuffer bb;
        master.saver()(send_b, bb);
        master.communicator().send(move_info.dst_proc, 0, bb.buffer);
    }
    else if (master.communicator().rank() == move_info.dst_proc)
    {
        recv_b = static_cast<Block*>(master.creator()());
        diy::MemoryBuffer bb;
        master.communicator().recv(move_info.src_proc, 0, bb.buffer);
        recv_b->load(recv_b, bb);
    }

    // move the link for the moving block from src to dst proc and update master on src and dst proc
    if (master.communicator().rank() == move_info.src_proc)
    {
        diy::Link* send_link = master.link(master.lid(move_info.move_gid));
        diy::MemoryBuffer bb;
        diy::LinkFactory::save(bb, send_link);
        master.communicator().send(move_info.dst_proc, 0, bb.buffer);

        // remove the block from the master
        Block* delete_block = static_cast<Block*>(master.get(master.lid(move_info.move_gid)));
        master.release(master.lid(move_info.move_gid));
        delete delete_block;
    }
    else if (master.communicator().rank() == move_info.dst_proc)
    {
        diy::MemoryBuffer bb;
        diy::Link* recv_link;
        master.communicator().recv(move_info.src_proc, 0, bb.buffer);
        recv_link = diy::LinkFactory::load(bb);

        // add block to the master
        master.add(move_info.move_gid, recv_b, recv_link);
    }

    // update the dynamic assigner
    if (master.communicator().rank() == move_info.src_proc)
        assigner.set_rank(move_info.dst_proc, move_info.move_gid, true);
}

int main(int argc, char* argv[])
{
    diy::mpi::environment     env(argc, argv);                          // diy equivalent of MPI_Init
    diy::mpi::communicator    world;                                    // diy equivalent of MPI communicator
    int                       bpr = 4;                                  // blocks per rank
    int                       iters = 1;                                // number of iterations to run
    int                       max_time = 1;                             // maximum time to compute a block (sec.)
    int                       method = 0;                               // dynamic load balancing method (-1: disable)
    double                    wall_time;                                // wall clock execution time for entire code
    bool                      help;

    using namespace opts;
    Options ops;
    ops
        >> Option('h', "help",      help,           "show help")
        >> Option('b', "bpr",       bpr,            "number of diy blocks per mpi rank")
        >> Option('i', "iters",     iters,          "number of iterations")
        >> Option('t', "max_time",  max_time,       "maximum time to compute a block (in seconds)")
        >> Option('m', "method",    method,         "dynamic load balancing method (-1: disable, 0: synchronous)")
        ;

    if (!ops.parse(argc,argv) || help)
    {
        if (world.rank() == 0)
        {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n";
            std::cout << "Tests work stealing\n";
            std::cout << ops;
        }
        return 1;
    }

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
//                              std::srand(gid * std::time(0));
                             std::srand(gid + 1);
                             b->work        = double(std::rand()) / RAND_MAX * WORK_MAX;

                             master.add(gid, b, l);
                         });

    // debug: display the decomposition
    master.foreach(&Block::show_block);

    world.barrier();
    wall_time = MPI_Wtime();

    // copy static assigner to dynamic assigner
    diy::DynamicAssigner    dynamic_assigner(world, world.size(), nblocks);
    set_dynamic_assigner(dynamic_assigner, master);                       // TODO: make a version of DynamicAssigner ctor take master and do this

    // this barrier is mandatory, do not remove
    // dynamic assigner needs to be fully updated and sync'ed across all procs before proceeding
    world.barrier();

    // debug: print the link for each block
//     for (auto i = 0; i < master.size(); i++)
//     {
//         Block*      b    = static_cast<Block*>(master.block(i));
//         diy::Link*  link = master.link(i);
//         fmt::print(stderr, "Link for block gid {} has size {}:\n", b->gid, link->size());
//         for (auto i = 0; i < link->size(); i++)
//             fmt::print(stderr, "[gid, proc] = [{}, {}] ", link->target(i).gid, link->target(i).proc);
//         fmt::print(stderr, "\n");
//     }

    std::vector<WorkInfo>   all_work_info(world.size());
    MoveInfo                move_info;

    // perform some iterative algorithm
    for (auto n = 0; n < iters; n++)
    {
        // some block computation
        master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
                { b->compute(cp, max_time, n); });

        if (method == -1)
            continue;

        // exchange info about work balance
        exchange_work_info(master, all_work_info);

        // decide what to move where
        if (!decide_move_info(master, all_work_info, move_info))
            continue;

        // move one block from src to dst proc
        move_block(dynamic_assigner, master, move_info);  // TODO: make this a dynamic assigner member function

        // debug: print the master of src and dst proc
        if (world.rank() == move_info.src_proc)
            fmt::print(stderr, "iteration {}: after moving gid {} from src rank {} to dst rank {}, src master size {}\n",
                    n, move_info.move_gid, move_info.src_proc, move_info.dst_proc, master.size());
        if (world.rank() == move_info.dst_proc)
            fmt::print(stderr, "iteration {}: after moving gid {} from src rank {} to dst rank {}, dst master size {}\n",
                    n, move_info.move_gid, move_info.src_proc, move_info.dst_proc, master.size());
        if (world.rank() == move_info.src_proc || world.rank() == move_info.dst_proc)
            for (auto i = 0; i < master.size(); i++)
                fmt::print(stderr, "lid {} gid {}\n", i, master.gid(i));

        // debug: print the link for each block
//         for (auto i = 0; i < master.size(); i++)
//         {
//             Block*      b    = static_cast<Block*>(master.block(i));
//             diy::Link*  link = master.link(i);
//             fmt::print(stderr, "Link for block gid {} has size {}:\n", b->gid, link->size());
//             for (auto i = 0; i < link->size(); i++)
//                 fmt::print(stderr, "[gid, proc] = [{}, {}] ", link->target(i).gid, link->target(i).proc);
//             fmt::print(stderr, "\n");
//         }
    }

    world.barrier();
    wall_time = MPI_Wtime() - wall_time;
    if (world.rank() == 0)
        fmt::print(stderr, "Total elapsed wall time {:.3} sec.\n", wall_time);
}
