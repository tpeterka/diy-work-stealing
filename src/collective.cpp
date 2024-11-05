#include <vector>
#include <iostream>
#include <bitset>

#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/master.hpp>
#include <diy/resolve.hpp>

#include "opts.h"
#include "common.hpp"

// exchange work information among all processes using synchronous collective method
// TODO: make a master member function
void exchange_work_info(const diy::Master&      master,
                        std::vector<Work>&      local_work,             // work for each local block TODO: eventually internal to master?
                        int                     iteration,              // current iteration (for debugging)
                        std::vector<WorkInfo>&  all_work_info)          // (output) global work info
{
    auto nlids  = master.size();                    // my local number of blocks
    auto nprocs = master.communicator().size();     // global number of procs

    WorkInfo my_work_info = { master.communicator().rank(), -1, 0, 0, (int)nlids };

    // compile my work info
    for (auto i = 0; i < master.size(); i++)
    {
        Block* block = static_cast<Block*>(master.block(i));
        my_work_info.proc_work += local_work[i];
        if (my_work_info.top_gid == -1 || my_work_info.top_work < local_work[i])
        {
            my_work_info.top_gid    = master.gid(i);
            my_work_info.top_work   = local_work[i];
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
// the user writes this function for now, TODO: implement in diy
void decide_move_info(const diy::Master&            master,
                      std::vector<WorkInfo>&        all_work_info,          // global work info
                      std::vector<MoveInfo>&        all_move_info)          // (output) move info for all moves
{

    all_move_info.clear();

#if 1       // move all blocks

    // sort all_work_info by proc_work
    std::sort(all_work_info.begin(), all_work_info.end(),
            [&](WorkInfo& a, WorkInfo& b) { return a.proc_work < b.proc_work; });

    // walk the all_work_info vector, shuffling heaviest blocks from heaviest procs to lightest procs
    // an approximation of the greedy list scheduling algorithm, stratified by process
    // instead of sorting all blocks, sort processes by their total work and then move the heaviest block among processes
    for (auto i = 0; i < all_work_info.size() / 2; i++)
    {
        int dst_idx = i;
        int src_idx = all_work_info.size() - i - 1;

        // check what the move would do to load balance between the two blocks
        int old_load_diff = abs(all_work_info[src_idx].proc_work - all_work_info[dst_idx].proc_work);
        int new_load_diff = abs(all_work_info[src_idx].proc_work - all_work_info[src_idx].top_work -
                               (all_work_info[dst_idx].proc_work + all_work_info[src_idx].top_work));

        // debug
//         fmt::print(stderr, "i {} src_proc {} dst_proc {} old_load_diff {} new_load_diff {}\n",
//                 i, all_work_info[src_idx].proc_rank, all_work_info[dst_idx].proc_rank, old_load_diff, new_load_diff);

        // don't make load balance worse and don't leave a proc with no blocks
        if (new_load_diff < old_load_diff && all_work_info[src_idx].nlids > 1)
        {
            MoveInfo move_info;
            move_info.src_proc  = all_work_info[src_idx].proc_rank;
            move_info.dst_proc  = all_work_info[dst_idx].proc_rank;
            move_info.move_gid  = all_work_info[src_idx].top_gid;
            all_move_info.push_back(move_info);
        }
    }
    return;

#else       // move one block

    // initialize move_info
    MoveInfo move_info = {-1, -1, -1};

//     fmt::print(stderr, "decide_move_info: move_info.move_gid {} move_info.src_proc {} move_info.dst_proc {}\n",
//             move_info.move_gid, move_info.src_proc, move_info.dst_proc);

    // parse all_work_info to decide block migration, all procs arriving at the same decision
    // for now pick the proc with the max. total work and move its top_gid to the proc with the min. total work
    // TODO later we can be more sophisticated, e.g., cut the work difference in half, etc., see related literature for guidance
    WorkInfo max_work = {-1, -1, 0, 0, 0};
    WorkInfo min_work = {-1, -1, 0, 0, 0};

    for (auto i = 0; i < all_work_info.size(); i++)                         // for all process ranks being considered (entire world or a sample)
    {
        // debug
//         fmt::print(stderr, "all_work_info[{}]: [{} {} {} {} {}]\n",
//             i, all_work_info[i].proc_rank, all_work_info[i].top_gid, all_work_info[i].top_work, all_work_info[i].proc_work, all_work_info[i].nlids);

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
        all_move_info.push_back(move_info);

        // debug
//         fmt::print(stderr, "decide_move_info(): move_gid {} src_proc {} dst_proc {}\n",
//                 move_info.move_gid, move_info.src_proc, move_info.dst_proc);
    }
    else
    {
        fmt::print(stderr, "decide_move_info(): nothing to move max_work.proc_rank {} min_work.proc_rank {} max_work.nlids {}\n",
                max_work.proc_rank, min_work.proc_rank, max_work.nlids);
    }

    return;

#endif

}

// move one block from src to dst proc
// TODO: make this a member function of dynamic assigner
void  move_block(diy::DynamicAssigner&   assigner,
                 diy::Master&            master,
                 const MoveInfo&         move_info,
                 int                     iteration)         // for debugging
{
    // debug
    if (master.communicator().rank() == move_info.src_proc)
        fmt::print(stderr, "iteration {}: moving gid {} from src rank {} to dst rank {}\n",
                iteration, move_info.move_gid, move_info.src_proc, move_info.dst_proc);

    // TP: this barrier may help to synchronize the information exchange before moving a block
    // TODO: decide whether it's needed on a production machine
//     master.communicator().barrier();

    // update the dynamic assigner
    if (master.communicator().rank() == move_info.src_proc)
        assigner.set_rank(move_info.dst_proc, move_info.move_gid, true);

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
}

// balance load using collective method
// TODO move into diy
void load_balance_collective(
        diy::Master&                master,             // diy master
        diy::DynamicAssigner&       dynamic_assigner,   // diy dynamic assigner
        std::vector<Work>&          local_work,         // work for each local block TODO: eventually internal to master?
        int                         iter)               // current iteration (for debugging)
{
    WorkInfo                my_work_info;               // my mpi process work info
    std::vector<WorkInfo>   all_work_info;              // work info collected from all mpi processes
    std::vector<MoveInfo>   all_move_info;              // move info for all moves

    // exchange info about load balance
    exchange_work_info(master, local_work, iter, all_work_info);

    // decide what to move where
    decide_move_info(master, all_work_info, all_move_info);

    // move blocks from src to dst proc
    for (auto i = 0; i < all_move_info.size(); i++)
        move_block(dynamic_assigner, master, all_move_info[i], iter);

    // fix links
    diy::fix_links(master, dynamic_assigner);
}

int main(int argc, char* argv[])
{
    diy::mpi::environment     env(argc, argv);                          // diy equivalent of MPI_Init
    diy::mpi::communicator    world;                                    // diy equivalent of MPI communicator
    int                       bpr = 4;                                  // blocks per rank
    int                       iters = 1;                                // number of iterations to run
    int                       max_time = 1;                             // maximum time to compute a block (sec.)
    double                    wall_time;                                // wall clock execution time for entire code
    bool                      help;

    using namespace opts;
    Options ops;
    ops
        >> Option('h', "help",          help,           "show help")
        >> Option('b', "bpr",           bpr,            "number of diy blocks per mpi rank")
        >> Option('i', "iters",         iters,          "number of iterations")
        >> Option('t', "max_time",      max_time,       "maximum time to compute a block (in seconds)")
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

//     diy::create_logger("trace");

    int                       nblocks = world.size() * bpr;             // total number of blocks in global domain
    diy::ContiguousAssigner   static_assigner(world.size(), nblocks);

    Bounds domain(3);                                                   // global data size
    domain.min[0] = domain.min[1] = domain.min[2] = 0;
    domain.max[0] = domain.max[1] = domain.max[2] = 255;

    // "main" master and decomposer for the actual blocks
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
//                              std::srand((gid + 1) * (master.communicator().rank() + 1));
                             std::srand(gid + 1);
                             b->work        = double(std::rand()) / RAND_MAX * WORK_MAX;

                             master.add(gid, b, l);
                         });

//     std::srand(world.rank() + 1);

//     int t = time(0);
//     fmt::print(stderr, "time {} rank {} time + rank {}\n", t, world.rank(), t + world.rank());
//     srand(t + world.rank());

    world.barrier();                                                    // barrier to synchronize clocks across procs, do not remove
    wall_time = MPI_Wtime();

    // copy static assigner to dynamic assigner
    diy::DynamicAssigner    dynamic_assigner(world, world.size(), nblocks);
    set_dynamic_assigner(dynamic_assigner, master);                       // TODO: make a version of DynamicAssigner ctor take master and do this

    // this barrier is mandatory, do not remove
    // dynamic assigner needs to be fully updated and sync'ed across all procs before proceeding
    world.barrier();

    // debug: print each block
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->show_block(cp); });

    // collect summary stats before beginning
    if (world.rank() == 0)
        fmt::print(stderr, "Summary stats before beginning\n");
    summary_stats(master);

    // perform some iterative algorithm
    for (auto n = 0; n < iters; n++)
    {
        // some block computation
        master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
                { b->compute(cp, max_time, n); });

        // compile my local work info
        std::vector<Work> local_work(master.size());
        for (auto i = 0; i < master.size(); i++)
            local_work[i] = static_cast<Block*>(master.block(i))->work;

        // synchronous collective load balancing
        load_balance_collective(master, dynamic_assigner, local_work, n);

        // debug: print the master of src and dst proc
//         if (world.rank() == move_info.src_proc)
//             fmt::print(stderr, "iteration {}: after moving gid {} from src rank {} to dst rank {}, src master size {}\n",
//                     n, move_info.move_gid, move_info.src_proc, move_info.dst_proc, master.size());
//         if (world.rank() == move_info.dst_proc)
//             fmt::print(stderr, "iteration {}: after moving gid {} from src rank {} to dst rank {}, dst master size {}\n",
//                     n, move_info.move_gid, move_info.src_proc, move_info.dst_proc, master.size());
//         if (world.rank() == move_info.src_proc || world.rank() == move_info.dst_proc)
//             for (auto i = 0; i < master.size(); i++)
//                 fmt::print(stderr, "lid {} gid {}\n", i, master.gid(i));
    }

    world.barrier();                                    // barrier to synchronize clocks over procs, do not remove
    wall_time = MPI_Wtime() - wall_time;
    if (world.rank() == 0)
        fmt::print(stderr, "Total elapsed wall time {:.3} sec.\n", wall_time);

    // load balance summary stats
    if (world.rank() == 0)
        fmt::print(stderr, "Summary stats upon completion\n");
    summary_stats(master);
}
