#include <vector>

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

// auxiliary empty block structure
struct AuxBlock
{
    static void*    create()            { return new AuxBlock; }
    static void     destroy(void* b)    { delete static_cast<AuxBlock*>(b); }
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

// compute summary stats on work information on root process
void stats_work_info(const diy::Master&         master,
                     std::vector<WorkInfo>&     all_work_info)
{
    Work tot_work = 0;
    Work max_work = 0;
    Work min_work = 0;
    float avg_work;
    float rel_imbalance;

    if (all_work_info.size())
    {
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
        {
            fmt::print(stderr, "Max process work {} Min process work {} Avg process work {} Rel process imbalance [(max - min) / max] {:.3}\n",
                    max_work, min_work, avg_work, rel_imbalance);
//             fmt::print(stderr, "Detailed list of all procs work:\n");
//             for (auto i = 0; i < all_work_info.size(); i++)
//                 fmt::print(stderr, "proc rank {} proc work {} top gid {} top gid work {}\n",
//                         all_work_info[i].proc_rank, all_work_info[i].proc_work, all_work_info[i].top_gid, all_work_info[i].top_work);
        }
    }
}

// gather summary stats on work information from all processes
void summary_stats(const diy::Master& master)
{
    std::vector<WorkInfo>   all_work_info;
    std::vector<Work> local_work(master.size());

    for (auto i = 0; i < master.size(); i++)
        local_work[i] = static_cast<Block*>(master.block(i))->work;

    gather_work_info(master, local_work, all_work_info);
    if (master.communicator().rank() == 0)
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



