#include <vector>
#include <iostream>
#include <bitset>

#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/master.hpp>
#include <diy/resolve.hpp>

#include "opts.h"

#define WORK_MAX    100                 // maximum work a block can have (in some user-defined units)

using Work = int;

typedef     diy::DiscreteBounds         Bounds;
typedef     diy::RegularGridLink        RGLink;

// information about a process' workload
struct WorkInfo
{
    int     proc_rank;          // mpi rank of this process
    int     top_gid;            // gid of most expensive block in this process TODO: can be top-k-gids, as long as k is fixed and known by all
    Work    top_work;           // work of top_gid TODO: can be vector of top-k work, as long as k is fixed and known by all
    Work    proc_work;          // total work of this process
    int     nlids;              // local number of blocks in this process
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
        diy::save(bb, b->work);
    }

    static void load(void* b_, diy::BinaryBuffer& bb)
    {
        Block* b = static_cast<Block*>(b_);

        diy::load(bb, b->gid);
        diy::load(bb, b->bounds);
        diy::load(bb, b->x);
        diy::load(bb, b->work);
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
        unsigned int usec = max_time * work * 10000L;

        // debug
//         fmt::print(stderr, "iteration {} block gid {} computing for {} s.\n", iter, gid, double(usec) / 1e6);

        usleep(usec);
    }

    // the block data
    int                 gid;
    Bounds              bounds;
    std::vector<double> x;                                              // some block data, e.g.
    Work                work;                                           // some estimate of how much work this block involves
};

// debug: print DynamicAssigner
void print_dynamic_assigner(const diy::Master&            master,
                            const diy::DynamicAssigner&   dynamic_assigner)
{
    fmt::print(stderr, "DynamicAssigner: ");
    for (auto i = 0; i < master.size(); i++)
        fmt::print(stderr, "[gid, proc] = [{}, {}] ", master.gid(i), dynamic_assigner.rank(master.gid(i)));
    fmt::print(stderr, "\n");
}

// debug: print the link for each block
void print_links(const diy::Master& master)
{
    for (auto i = 0; i < master.size(); i++)
    {
        Block*      b    = static_cast<Block*>(master.block(i));
        diy::Link*  link = master.link(i);
        fmt::print(stderr, "Link for gid {} is size {}: ", b->gid, link->size());
        for (auto i = 0; i < link->size(); i++)
            fmt::print(stderr, "[gid, proc] = [{}, {}] ", link->target(i).gid, link->target(i).proc);
        fmt::print(stderr, "\n");
    }
}

// exchange work information among all processes using synchronous collective method
// TODO: make a master member function
void exchange_work_info(const diy::Master&      master,
                        std::vector<Work>&      local_work,             // work for each local block TODO: eventually internal to master?
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

// get work information from a random sample of processes
// TODO: make a master member function
void exchange_sample_work_info(const diy::Master&       master,
                               std::vector<Work>&       local_work,             // work for each local block TODO: eventually internal to master?
                               float                    sample_frac,            // fraction of procs to sample 0.0 < sample_size <= 1.0
                               int                      iter,                   // current iteration
                               WorkInfo&                my_work_info,           // (output) my work info
                               std::vector<WorkInfo>&   sample_work_info)       // (output) vector of sorted sample work info, sorted by increasing total work per process
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
    int nsamples = sample_frac * nprocs;
    std::set<int> sample_procs;
    for (auto i = 0; i < nsamples; i++)
    {
        std::srand((iter + 1) * (i + 1) * (my_proc + 1));
        int rand_proc;
        do
        {
            rand_proc = double(std::rand()) / RAND_MAX * master.communicator().size();
        } while (sample_procs.find(rand_proc) != sample_procs.end() || rand_proc == my_proc);
        sample_procs.insert(rand_proc);
    }

    // debug
//     fmt::print(stderr, "sample_procs [{}]\n", fmt::join(sample_procs, ","));

    // communication protocol based on diy's rexchange, which is based on NBX algorithm of Hoefler et al.,
    // Scalable Communication Protocols for Dynamic Sparse Data Exchange, 2010.

    // send requests for work info to sample_procs using tag 0
    std::vector<diy::mpi::request> reqs(sample_procs.size());
    int a = 1;                      // simple integer to request work info
    int i = 0;
    for (auto proc_iter = sample_procs.begin(); proc_iter != sample_procs.end(); proc_iter++)
    {
        reqs[i] = master.communicator().issend(*proc_iter, 0, a);
        i++;
    }

    // receive requets for work info using tag 0
    bool done = false;
    bool barrier_active = false;
    diy::mpi::request barrier_req;
    std::vector<int> req_procs;     // requests for work info received from these processes
    diy::mpi::optional<diy::mpi::status> status;
    while (!done)
    {
        // nudge mpi's progress engine
        for (auto i = 0; i < sample_procs.size(); i++)
            reqs[i].test();

        status = master.communicator().iprobe(diy::mpi::any_source, 0);
        while (status)
        {
            int req;
            master.communicator().recv(status->source(), 0, req);
            req_procs.push_back(status->source());

            // check for more requests
            status = master.communicator().iprobe(diy::mpi::any_source, 0);
        }

        // synchronize
        if (barrier_active)
        {
            if (barrier_req.test())
                done = true;
        }
        else
        {
            barrier_req = master.communicator().ibarrier();
            barrier_active = true;
        }
    }

    // sanity check: wait for all requests to finish TODO: necessary?
//     for (auto i = 0; i < reqs.size(); i++)
//         reqs[i].wait();
//     barrier_req.wait();

    // send work info using tag 1
    std::vector<diy::mpi::request> reqs1;
    for (auto i = 0; i < req_procs.size(); i++)
        reqs1.push_back(master.communicator().issend(req_procs[i], 1, my_work_info_vec));

    // receive work info using tag 1
    std::vector<int>   other_work_info_vec(5);
    sample_work_info.resize(nsamples);
    i = 0;
    done = false;
    barrier_active = false;
    while (!done)
    {
        // nudge mpi's progress engine
        for (auto i = 0; i < reqs1.size(); i++)
            reqs1[i].test();

        status = master.communicator().iprobe(diy::mpi::any_source, 1);
        while (status)
        {
            master.communicator().recv(status->source(), 1, other_work_info_vec);

            sample_work_info[i].proc_rank = other_work_info_vec[0];
            sample_work_info[i].top_gid   = other_work_info_vec[1];
            sample_work_info[i].top_work  = other_work_info_vec[2];
            sample_work_info[i].proc_work = other_work_info_vec[3];
            sample_work_info[i].nlids     = other_work_info_vec[4];
            i++;

            // check for more requests
            status = master.communicator().iprobe(diy::mpi::any_source, 1);
        }

        // synchronize
        if (barrier_active)
        {
            if (barrier_req.test())
                done = true;
        }
        else
        {
            barrier_req = master.communicator().ibarrier();
            barrier_active = true;
        }
    }

    // sanity check: wait for all send requests to finish TODO: necessary?
//     for (auto i = 0; i < reqs1.size(); i++)
//         reqs1[i].wait();
//     barrier_req.wait();

    // sort sample_work_info by proc_work
    struct
    {
        bool operator()(WorkInfo& a, WorkInfo& b) const { return a.proc_work < b.proc_work; }
    }
    CompareProcWork;
    std::sort(sample_work_info.begin(), sample_work_info.end(), CompareProcWork);

    // debug
//     for (auto i = 0; i < sample_work_info.size(); i++)
//         fmt::print(stderr, "sample_work_info[{}]: proc_rank {} top_gid {} top_work {} proc_work {} nlids {}\n",
//                 i, sample_work_info[i].proc_rank, sample_work_info[i].top_gid, sample_work_info[i].top_work, sample_work_info[i].proc_work, sample_work_info[i].nlids);
}

// determine move info from work info
// the user writes this function for now, TODO: implement in diy
void decide_move_info(const diy::Master&            master,
                      const std::vector<WorkInfo>&  all_work_info,          // global work info
                      MoveInfo&                     move_info)              // (output) move info
{
    // initialize move_info
    move_info = {-1, -1, -1};

//     fmt::print(stderr, "decide_move_info: move_info.move_gid {} move_info.src_proc {} move_info.dst_proc {}\n",
//             move_info.move_gid, move_info.src_proc, move_info.dst_proc);

    // if my proc is not included in the global work info, nothing to do
    if (!all_work_info.size())
        return;

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
}

// determine move info from sampled work info
// the user writes this function for now, TODO: implement in diy
void decide_sample_move_info(const diy::Master&              master,
                             const std::vector<WorkInfo>&    sample_work_info,           // sampled work info
                             const WorkInfo&                 my_work_info,               // my work info
                             std::vector<MoveInfo>&          multi_move_info)            // (output) move info
{
    // initialize move_info, multi_move_info
    MoveInfo move_info = {-1, -1, -1};
    multi_move_info.clear();

    // pick a cutoff quantile above which to move blocks, see if my rank qualifies
    float quantile = 0.8;                                                               // TODO: tune this, or make a user parameter
    int my_work_idx = sample_work_info.size();                                          // index where my work would be in the sample_work
    for (auto i = 0; i < sample_work_info.size(); i++)
    {
        if (my_work_info.proc_work < sample_work_info[i].proc_work)
        {
            my_work_idx = i;
            break;
        }
    }

    // debug
//     fmt::print(stderr, "decide_sample_move_info(): my_work {} my_work_idx {}\n",
//             my_work_info.proc_work, my_work_idx);

    // communication protocol based on diy's rexchange, which is based on NBX algorithm of Hoefler et al.,
    // Scalable Communication Protocols for Dynamic Sparse Data Exchange, 2010.

    // my mpi process is either a donor of work or a potentially a recipient, but not both

    // donor
    if (my_work_idx >= quantile * sample_work_info.size())
    {

        // pick the destination process to be the mirror image of my work location in the samples
        // ie, the heavier my process, the lighter the destination process
        int target = sample_work_info.size() - my_work_idx;

        // don't run out of blocks
        if (my_work_info.nlids > 1)
        {
            move_info.move_gid = my_work_info.top_gid;
            move_info.src_proc = my_work_info.proc_rank;
            move_info.dst_proc = sample_work_info[target].proc_rank;
            multi_move_info.push_back(move_info);

            // debug
//             fmt::print(stderr, "decide_sample_move_info(): my_work {} move_gid {} src_proc {} dst_proc {}\n",
//                     my_work_info.proc_work, move_info.move_gid, move_info.src_proc, move_info.dst_proc);
        }

        // send move_info to destination proc using tag 2
        std::vector<int> move_info_vec =
        {
            move_info.move_gid,
            move_info.src_proc,
            move_info.dst_proc
        };
        diy::mpi::request request, barrier_req;
        request = master.communicator().issend(move_info.dst_proc, 2, move_info_vec);
        request.test();                 // nudge mpi's progress engine
        barrier_req = master.communicator().ibarrier();

        // sanity check: wait for send request to finish TODO: necessary?
//         request.wait();
//         barrier_req.wait();

        // debug
//         fmt::print(stderr, "sending move info to proc {}\n", move_info.dst_proc);
    }

    // potential recipient
    else
    {
        // check for move_info messages using tag 2
        bool done = false;
        bool barrier_active = false;
        diy::mpi::request barrier_req;
        diy::mpi::optional<diy::mpi::status> status;
        std::vector<int> move_info_vec(3);

        while (!done)
        {
            status = master.communicator().iprobe(diy::mpi::any_source, 2);
            while (status)
            {
                master.communicator().recv(status->source(), 2, move_info_vec);
                move_info.move_gid = move_info_vec[0];
                move_info.src_proc = move_info_vec[1];
                move_info.dst_proc = move_info_vec[2];
                multi_move_info.push_back(move_info);

                // debug
//                 fmt::print(stderr, "recvd move info from proc {}: move_gid {} src_proc {} dst_proc {}\n",
//                     status->source(), move_info.move_gid, move_info.src_proc, move_info.dst_proc);

                // check for more requests
                status = master.communicator().iprobe(diy::mpi::any_source, 2);
            }

            // synchronize
            if (barrier_active)
            {
                if (barrier_req.test())
                    done = true;
            }
            else
            {
                barrier_req = master.communicator().ibarrier();
                barrier_active = true;
            }
        }

        // initialize to -1's if no move_info messages were received
        if (!multi_move_info.size())
            multi_move_info.push_back(move_info);

        // sanity check, wait for all requests to finish TODO: necessary?
//         barrier_req.wait();
    }

    return;
}

// gather work information from all processes in order to collect summary stats
void   gather_work_info(const diy::Master&      master,
                        std::vector<Work>&      local_work,             // work for each local block TODO: eventually internal to master?
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

    // gather work info
    diy::mpi::gather(master.communicator(), my_work_info_vec, all_work_info_vec, 0);

    // unpack received info into vector of structs
    if (master.communicator().rank() == 0)
    {
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
}

// compute summary stats on work information from all processes
void stats_work_info(const diy::Master&         master,
                     std::vector<WorkInfo>&     all_work_info)
{
    Work tot_work = 0;
    Work max_work = 0;
    Work min_work = 0;
    float avg_work;
    float rel_imbalance;

    for (auto i = 0; i < all_work_info.size(); i++)
    {
        if (i == 0 || all_work_info[i].proc_work < min_work)
            min_work = all_work_info[i].proc_work;
        if (i == 0 || all_work_info[i].proc_work > max_work)
            max_work = all_work_info[i].proc_work;
        tot_work += all_work_info[i].proc_work;
    }

    avg_work = tot_work / all_work_info.size();
    rel_imbalance   = float(max_work - min_work) / max_work;

    if (master.communicator().rank() == 0)
    fmt::print(stderr, "Max process work {} Min process work {} Avg process work {} Rel process imbalance [(max - min) / max] {:.3}\n",
            max_work, min_work, avg_work, rel_imbalance);
}

void summary_stats(const diy::Master& master)
{
    std::vector<WorkInfo>   all_work_info;
    std::vector<Work> local_work(master.size());

    for (auto i = 0; i < master.size(); i++)
        local_work[i] = static_cast<Block*>(master.block(i))->work;

    gather_work_info(master, local_work, all_work_info);
    stats_work_info(master, all_work_info);
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
                 const MoveInfo&         move_info,
                 int                     iteration)         // for debugging
{
    // debug
//     fmt::print(stderr, "move_block(): iteration {} move_info move_gid {} src_proc {} dst_proc {}\n",
//             iteration, move_info.move_gid, move_info.src_proc, move_info.dst_proc);

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

int main(int argc, char* argv[])
{
    diy::mpi::environment     env(argc, argv);                          // diy equivalent of MPI_Init
    diy::mpi::communicator    world;                                    // diy equivalent of MPI communicator
    int                       bpr = 4;                                  // blocks per rank
    int                       iters = 1;                                // number of iterations to run
    int                       max_time = 1;                             // maximum time to compute a block (sec.)
    int                       method = 0;                               // dynamic load balancing method (-1: disable)
    float                     sample_frac = 0.5;                        // fraction of world procs to sample
    double                    wall_time;                                // wall clock execution time for entire code
    bool                      help;

    using namespace opts;
    Options ops;
    ops
        >> Option('h', "help",          help,           "show help")
        >> Option('b', "bpr",           bpr,            "number of diy blocks per mpi rank")
        >> Option('i', "iters",         iters,          "number of iterations")
        >> Option('t', "max_time",      max_time,       "maximum time to compute a block (in seconds)")
        >> Option('m', "method",        method,         "dynamic load balancing method (-1: disable, 0: synchronous collective, 1: sampling)")
        >> Option('s', "sample_frac",   sample_frac,    "fraction of world procs to sample")
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
                             std::srand(gid + 1);
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

        // no load balancing
        if (method == -1)
            continue;

        // synchronous collective method
        else if (method == 0)
        {
            // exchange info about load balance
            exchange_work_info(master, local_work, all_work_info);
            // decide what to move where (one block move currently, TODO: more than one block)
            decide_move_info(master, all_work_info, move_info);
            // move one block from src to dst proc TODO: move more than one block
            move_block(dynamic_assigner, master, move_info, n);  // TODO: make this a dynamic assigner member function
            // fix links
            diy::fix_links(master, dynamic_assigner);

            // debug
            if (world.rank() == move_info.src_proc)
                fmt::print(stderr, "iteration {}: moving gid {} from src rank {} to dst rank {}\n",
                        n, move_info.move_gid, move_info.src_proc, move_info.dst_proc);
        }

        // sampling method
        else if (method == 1)
        {
            // exchange info about load balance
            exchange_sample_work_info(master, local_work, sample_frac, n, my_work_info, sample_work_info);
            // decide what to move where
            decide_sample_move_info(master, sample_work_info, my_work_info, multi_move_info);
            // move one blocks from src to dst proc
            for (auto i = 0; i < multi_move_info.size(); i++)
            {
                move_block(dynamic_assigner, master, multi_move_info[i], n);  // TODO: make this a dynamic assigner member function

                // debug
                if (world.rank() == multi_move_info[i].src_proc)
                    fmt::print(stderr, "iteration {}: moving gid {} from src rank {} to dst rank {}\n",
                            n, multi_move_info[i].move_gid, multi_move_info[i].src_proc, multi_move_info[i].dst_proc);
            }
            // fix links
            diy::fix_links(master, dynamic_assigner);

        }

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
