#include <vector>
#include <random>
#include <time.h>

#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/master.hpp>
#include <diy/resolve.hpp>

#include "opts.h"
#include "common.hpp"

// send requests for work info
void send_req(Block* b,                                 // local block
              const diy::Master::ProxyWithLink& cp,     // communication proxy for neighbor blocks
              std::set<int>& procs)                     // processes to query
{
    // send requests for work info to sample_procs
    int v = 1;                                          // any message will do
    for (auto proc_iter = procs.begin(); proc_iter != procs.end(); proc_iter++)
    {
        int gid    = *proc_iter;
        int proc   = *proc_iter;
        diy::BlockID dest_block = {gid, proc};
        cp.enqueue(dest_block, v);
    }
}

// receive requests for work info
void recv_req(Block* b,                                 // local block
              const diy::Master::ProxyWithLink& cp,     // communication proxy for neighbor blocks
              std::vector<int>& req_procs)              // processes requesting work info
{
    std::vector<int> incoming_gids;
    cp.incoming(incoming_gids);

    // for anything incoming, dequeue data received in the last exchange
    for (int i = 0; i < incoming_gids.size(); i++)
    {
        int gid = incoming_gids[i];
        if (cp.incoming(gid).size())
        {
            int v;
            cp.dequeue(gid, v);
            req_procs.push_back(gid);                   // aux_master has 1 gid per proc, so gid = proc
        }
    }
}

// get work information from a random sample of processes
// TODO: make a master member function
void exchange_sample_work_info(diy::Master&             master,                 // the real master with multiple blocks per process
                               diy::Master&             aux_master,             // auxiliary master with 1 block per process for communicating between procs
                               std::vector<Work>&       local_work,             // work for each local block TODO: eventually internal to master?
                               float                    sample_frac,            // fraction of procs to sample 0.0 < sample_size <= 1.0
                               int                      iter,                   // current iteration
                               WorkInfo&                my_work_info,           // (output) my work info
                               std::vector<WorkInfo>&   sample_work_info,       // (output) vector of sorted sample work info, sorted by increasing total work per process
                               std::mt19937&            gen)                    // random number generator TODO: include in diy
{
    auto nlids  = master.size();                    // my local number of blocks
    auto nprocs = master.communicator().size();     // global number of procs
    auto my_proc = master.communicator().rank();    // rank of my proc

    // compile my work info
    my_work_info = { master.communicator().rank(), -1, 0, 0, (int)nlids };
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

    // pick a random sample of processes, w/o duplicates, and excluding myself
    int nsamples = sample_frac * (nprocs - 1);
    std::set<int> sample_procs;
    for (auto i = 0; i < nsamples; i++)
    {
        int rand_proc;
        do
        {
            std::uniform_int_distribution<> distrib(0, master.communicator().size() - 1);
            rand_proc = distrib(gen);
        } while (sample_procs.find(rand_proc) != sample_procs.end() || rand_proc == my_proc);
        sample_procs.insert(rand_proc);
    }

    // debug
//     fmt::print(stderr, "sample_procs [{}]\n", fmt::join(sample_procs, ","));

    // rexchange requests for work info
    std::vector<int> req_procs;     // requests for work info received from these processes
    aux_master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { send_req(b, cp, sample_procs); });
    aux_master.exchange(true);      // true = remote
    aux_master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { recv_req(b, cp, req_procs); });

    // debug
//     fmt::print(stderr, "req_procs [{}]\n", fmt::join(req_procs, ","));

    // send work info
    std::vector<diy::mpi::request> reqs(req_procs.size());
    for (auto i = 0; i < req_procs.size(); i++)
        reqs[i] = master.communicator().isend(req_procs[i], 0, my_work_info_vec);

    // receive work info
    int i = 0;
    std::vector<int>   other_work_info_vec(5);
    sample_work_info.resize(nsamples);
    for (auto proc_iter = sample_procs.begin(); proc_iter != sample_procs.end(); proc_iter++)
    {
        master.communicator().recv(diy::mpi::any_source, 0, other_work_info_vec);
        sample_work_info[i].proc_rank = other_work_info_vec[0];
        sample_work_info[i].top_gid   = other_work_info_vec[1];
        sample_work_info[i].top_work  = other_work_info_vec[2];
        sample_work_info[i].proc_work = other_work_info_vec[3];
        sample_work_info[i].nlids     = other_work_info_vec[4];
        i++;
    }

    // ensure all the send requests cleared
    for (auto i = 0; i < req_procs.size(); i++)
        reqs[i].wait();

    // sort sample_work_info by proc_work
    std::sort(sample_work_info.begin(), sample_work_info.end(),
            [&](WorkInfo& a, WorkInfo& b) { return a.proc_work < b.proc_work; });

    // debug
//     for (auto i = 0; i < sample_work_info.size(); i++)
//         fmt::print(stderr, "sample_work_info[{}]: proc_rank {} top_gid {} top_work {} proc_work {} nlids {}\n",
//                 i, sample_work_info[i].proc_rank, sample_work_info[i].top_gid, sample_work_info[i].top_work, sample_work_info[i].proc_work, sample_work_info[i].nlids);
}

// send block
void send_block(Block*                              b,                  // local block
                const diy::Master::ProxyWithLink&   cp,                 // communication proxy for neighbor blocks
                diy::Master&                        master,             // real master with multiple blocks per process
                diy::DynamicAssigner&               dynamic_assigner,   // dynamic assigner
                const std::vector<WorkInfo>&        sample_work_info,   // sampled work info
                const WorkInfo&                     my_work_info,       // my work info
                float                               quantile,           // quantile cutoff above which to move blocks (0.0 - 1.0)
                int                                 iter)               // current iteration number (for debugging)
{
    MoveInfo move_info = {-1, -1, -1};

    // my rank's position in the sampled work info, sorted by proc_work
    int my_work_idx = sample_work_info.size();                                          // index where my work would be in the sample_work
    for (auto i = 0; i < sample_work_info.size(); i++)
    {
        if (my_work_info.proc_work < sample_work_info[i].proc_work)
        {
            my_work_idx = i;
            break;
        }
    }

    // send my heaviest block if it passes the quantile cutoff and I won't run out of blocks
    if (my_work_idx >= quantile * sample_work_info.size() && master.size() > 1)
    {
        // pick the destination process to be the mirror image of my work location in the samples
        // ie, the heavier my process, the lighter the destination process
        int target = sample_work_info.size() - my_work_idx;

        move_info.move_gid = my_work_info.top_gid;
        move_info.src_proc = my_work_info.proc_rank;
        move_info.dst_proc = sample_work_info[target].proc_rank;

        // debug
        //             fmt::print(stderr, "decide_sample_move_info(): my_work {} move_gid {} src_proc {} dst_proc {}\n",
        //                     my_work_info.proc_work, move_info.move_gid, move_info.src_proc, move_info.dst_proc);

        // update the dynamic assigner
        dynamic_assigner.set_rank(move_info.dst_proc, move_info.move_gid, true);

        // destination in aux_master, where gid = proc
        diy::BlockID dest_block = {move_info.dst_proc, move_info.dst_proc};

        // enqueue the gid of the moving block
        cp.enqueue(dest_block, move_info.move_gid);

        // enqueue the block
        void* send_b;
        send_b = master.block(master.lid(move_info.move_gid));
        diy::MemoryBuffer bb;
        master.saver()(send_b, bb);
        cp.enqueue(dest_block, bb.buffer);

        // enqueue the link for the block
        diy::Link* send_link = master.link(master.lid(move_info.move_gid));
        diy::LinkFactory::save(bb, send_link);
        cp.enqueue(dest_block, bb.buffer);

        // remove the block from the master
        Block* delete_block = static_cast<Block*>(master.get(master.lid(move_info.move_gid)));
        master.release(master.lid(move_info.move_gid));
        delete delete_block;

        // debug
        if (master.communicator().rank() == move_info.src_proc)
            fmt::print(stderr, "iteration {}: moving gid {} from src rank {} to dst rank {}\n",
                    iter, move_info.move_gid, move_info.src_proc, move_info.dst_proc);
    }
}

// receive block
void recv_block(Block* b,                                       // local block
                const diy::Master::ProxyWithLink&   cp,         // communication proxy for neighbor blocks
                diy::Master&                        master)     // real master with multiple blocks per process
{
    std::vector<int> incoming_gids;
    cp.incoming(incoming_gids);

    Block* recv_b;

    // for anything incoming, dequeue data received in the last exchange
    for (int i = 0; i < incoming_gids.size(); i++)
    {
        int gid = incoming_gids[i];
        if (cp.incoming(gid).size())
        {
            // dequeue the gid of the moving block
            int move_gid;
            cp.dequeue(gid, move_gid);

            // dequeue the block
            recv_b = static_cast<Block*>(master.creator()());
            diy::MemoryBuffer bb;
            cp.dequeue(gid, bb.buffer);
            recv_b->load(recv_b, bb);

            // dequeue the link
            diy::Link* recv_link;
            cp.dequeue(gid, bb.buffer);
            recv_link = diy::LinkFactory::load(bb);

            // add block to the master
            master.add(move_gid, recv_b, recv_link);
        }
    }
}

// move blocks based on sampled work info
void move_sample_blocks(diy::Master&                    master,                 // real master with multiple blocks per process
                        diy::Master&                    aux_master,             // auxiliary master with 1 block per process for communcating between procs
                        diy::DynamicAssigner&           dynamic_assigner,       // dynamic assigner
                        const std::vector<WorkInfo>&    sample_work_info,       // sampled work info
                        const WorkInfo&                 my_work_info,           // my work info
                        float                           quantile,               // quantile cutoff above which to move blocks (0.0 - 1.0)
                        int                             iter)                   // current iteration (for debugging)
{
    // rexchange moving blocks
    aux_master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { send_block(b, cp, master, dynamic_assigner, sample_work_info, my_work_info, quantile, iter); });
    aux_master.exchange(true);      // true = remote
    aux_master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { recv_block(b, cp, master); });
}

// balance load using sampling method
// TODO move into diy
void load_balance_sampling(
        diy::Master&                master,
        diy::StaticAssigner&        static_assigner,    // diy static assigner
        diy::DynamicAssigner&       dynamic_assigner,   // diy dynamic assigner
        std::vector<Work>&          local_work,         // work for each local block TODO: eventually internal to master?
        float                       sample_frac,        // fraction of procs to sample 0.0 < sample_size <= 1.0
        float                       quantile,           // quantile cutoff above which to move blocks (0.0 - 1.0)
        int                         iter,               // current iteration (for debugging)
        std::mt19937&               gen)                // random number generator TODO: include in diy
{
    WorkInfo                my_work_info;               // my mpi process work info
    std::vector<WorkInfo>   sample_work_info;           // work info collecting from sampling other mpi processes
    std::vector<MoveInfo>   multi_move_info;            // move info for moving multiple blocks

    // "auxiliary" master and decomposer for using rexchange for load balancing, 1 block per process
    Bounds domain(1);                                   // any fake domain
    domain.min[0] = 0;
    domain.max[0] = master.communicator().size() + 1;
    diy::Master                     aux_master(master.communicator(), 1, -1, &Block::create, &Block::destroy);
    diy::ContiguousAssigner         aux_assigner(master.communicator().size(), master.communicator().size());
    diy::RegularDecomposer<Bounds>  aux_decomposer(1, domain, master.communicator().size());
    aux_decomposer.decompose(master.communicator().rank(), aux_assigner, aux_master);

    // exchange info about load balance
    exchange_sample_work_info(master, aux_master, local_work, sample_frac, iter, my_work_info, sample_work_info, gen);

    // move blocks
    move_sample_blocks(master, aux_master, dynamic_assigner, sample_work_info, my_work_info, quantile, iter);

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
    float                     sample_frac = 0.5;                        // fraction of world procs to sample (0.0 - 1.0)
    float                     quantile = 0.8;                           // quantile cutoff above which to move blocks (0.0 - 1.0)
    double                    wall_time;                                // wall clock execution time for entire code
    bool                      help;

    using namespace opts;
    Options ops;
    ops
        >> Option('h', "help",          help,           "show help")
        >> Option('b', "bpr",           bpr,            "number of diy blocks per mpi rank")
        >> Option('i', "iters",         iters,          "number of iterations")
        >> Option('t', "max_time",      max_time,       "maximum time to compute a block (in seconds)")
        >> Option('s', "sample_frac",   sample_frac,    "fraction of world procs to sample (0.0 - 1.0)")
        >> Option('q', "quantile",      quantile,       "quantile cutoff above which to move blocks (0.0 - 1.0)")
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

    // seed random number generator for diy, broadcast seed, offset by rank
    // TODO: move this into diy
    std::random_device rd;                      // seed source for the random number engine
    uint s = rd();
    diy::mpi::broadcast(world, s, 0);
    std::mt19937 gen(s + world.rank());         // mersenne_twister_engine

    // seed random number generator for user code, broadcast seed, offset by rank
    time_t t;
    if (world.rank() == 0)
        t = time(0);
    diy::mpi::broadcast(world, t, 0);
    srand(t + world.rank());

    // create master for managing blocks in this process
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
                             b->work        = double(std::rand()) / RAND_MAX * WORK_MAX;
                             master.add(gid, b, l);
                         });

    world.barrier();                                                    // barrier to synchronize clocks across procs, do not remove
    wall_time = MPI_Wtime();

    // copy static assigner to dynamic assigner
    diy::DynamicAssigner    dynamic_assigner(world, world.size(), nblocks);
    set_dynamic_assigner(dynamic_assigner, master);                       // TODO: make a version of DynamicAssigner ctor take master and do this

    // this barrier is mandatory, do not remove
    // dynamic assigner needs to be fully updated and sync'ed across all procs before proceeding
    world.barrier();

    WorkInfo                my_work_info;
    std::vector<WorkInfo>   all_work_info;
    std::vector<WorkInfo>   sample_work_info;
    MoveInfo                move_info;
    std::vector<MoveInfo>   multi_move_info;

    // debug: print each block
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//             { b->show_block(cp); });

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

        // sampling load balancing method
        load_balance_sampling(master, static_assigner, dynamic_assigner, local_work, sample_frac, quantile, n, gen);
    }

    // debug: print the master
//     for (auto i = 0; i < master.size(); i++)
//         fmt::print(stderr, "lid {} gid {}\n", i, master.gid(i));

    world.barrier();                                    // barrier to synchronize clocks over procs, do not remove
    wall_time = MPI_Wtime() - wall_time;
    if (world.rank() == 0)
        fmt::print(stderr, "Total elapsed wall time {:.3} sec.\n", wall_time);

    // load balance summary stats
    if (world.rank() == 0)
        fmt::print(stderr, "Summary stats upon completion\n");
    summary_stats(master);
}
