// A Qwen2.5 layer frontend and compute-only scheduler. Tile service times come
// from executing the original COCOSSim units, not from a FLOP/peak estimate.
// With no shared memory timing, a job's service time is independent of its
// co-runners: caching these times permits an equivalent discrete-event replay.
#include "units/standard/SysArray.h"
#include "units/standard/VectorUnit.h"
#include <array>
#include <cassert>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>

using U = uint64_t;
constexpr int SA = 0, VPU = 1;
constexpr int SIDE = 32, LANES = 128, ROW_CHUNK = 128, EW_CHUNK = 16384;
struct Timing { U cycles = 0, work = 0; };
Timing operator+(Timing a, Timing b) { return {a.cycles + b.cycles, a.work + b.work}; }
void require(bool good, const std::string &why) { if (!good) throw std::runtime_error(why); }

struct Oracle {
  std::map<std::string, Timing> cache;
  template <class S, class J> Timing measure(S &s, J &j) {
    s.j = &j;
    s.init();
    int idle = 0, idle_types[2] = {};
    gcycles = 0;
    while (!j.is_done) {
      ++gcycles;
      s.increment([](Job *) { throw std::runtime_error("oracle job has a child"); }, idle, idle_types);
      require(gcycles < 100000000, "native tile failed to complete");
    }
    require(to_enqueue.empty() && !s.is_idle_from_memory, "memory activity in compute-only backend");
    return {gcycles, s.total_work};
  }
  Timing sa(int m, int k, int n) {
    std::string key = "SA:" + std::to_string(m) + ":" + std::to_string(k) + ":" + std::to_string(n);
    auto it = cache.find(key); if (it != cache.end()) return it->second;
    require(m > 0 && m <= SIDE && n > 0 && n <= SIDE && k > 0, "invalid SA tile");
    SystolicArray::SysArrayJob j(m, k, n, SIDE, false);
    SystolicArray::SysArrayState s(SIDE, false);
    Timing t = measure(s, j);
    require(t.work == U(m) * k * n, "native SA work mismatch");
    return cache[key] = t;
  }
  Timing vec(int lin, int par, int cost, bool reduce = false) {
    std::string key = std::string(reduce ? "R:" : "B:") + std::to_string(lin) + ":" + std::to_string(par) + ":" + std::to_string(cost);
    auto it = cache.find(key); if (it != cache.end()) return it->second;
    require(lin > 0 && par > 0 && cost > 0, "invalid vector tile");
    auto phase = reduce ? VectorUnit::REDUCE : VectorUnit::BROADCAST;
    std::vector<std::pair<VectorUnit::VPUPhase, int>> phases{{phase, cost}};
    VectorUnit::VecUnitJob j(lin, par, true, phases, 1, true);
    VectorUnit::VecUnitState s(LANES);
    Timing t = measure(s, j);
    require(t.work == U(lin) * par * cost, "native VPU work mismatch");
    return cache[key] = t;
  }
  Timing rms(int dim, int rows, int nonlinear) {
    // square; sum; multiply by 1/D, add epsilon, rsqrt; normalize and weight.
    return vec(dim, rows, 1) + vec(dim, rows, 1, true)
         + vec(1, rows, 2 + nonlinear) + vec(dim, rows, 2);
  }
  Timing softmax(int seq, int rows, int nonlinear) {
    // Scale and add causal mask; max; subtract and exp; sum; reciprocal; mul.
    return vec(seq, rows, 2) + vec(seq, rows, 1, true)
         + vec(seq, rows, 1 + nonlinear) + vec(seq, rows, 1, true)
         + vec(1, rows, nonlinear) + vec(seq, rows, 1);
  }
};

struct Batch { U count; Timing timing; };
struct Stage {
  std::string name, category;
  int type;
  std::vector<Batch> batches;
  std::vector<int> deps, children;
  U count() const { U n = 0; for (auto &b: batches) n += b.count; return n; }
};
struct Graph {
  std::vector<Stage> stages;
  int add(std::string name, std::string category, int type, std::vector<Batch> batches, std::vector<int> deps = {}) {
    std::sort(deps.begin(), deps.end());
    deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
    int id = stages.size();
    stages.push_back({name, category, type, batches, deps, {}});
    require(stages.back().count() > 0, "empty stage " + name);
    for (int d: deps) {
      require(d >= 0 && d < id, "graph must be topologically constructed");
      stages[d].children.push_back(id);
    }
    return id;
  }
  std::array<U, 2> work() const {
    std::array<U, 2> ans{};
    for (auto &s: stages) for (auto &b: s.batches) ans[s.type] += b.count * b.timing.work;
    return ans;
  }
  std::array<U, 2> busy() const {
    std::array<U, 2> ans{};
    for (auto &s: stages) for (auto &b: s.batches) ans[s.type] += b.count * b.timing.cycles;
    return ans;
  }
};
std::vector<std::pair<int,U>> chunks(int size, int tile) {
  std::vector<std::pair<int,U>> a;
  if (size / tile) a.push_back({tile, U(size / tile)});
  if (size % tile) a.push_back({size % tile, 1});
  return a;
}
struct Layer {
  Oracle &o;
  Graph g;
  int batch, seq, nonlinear;
  static constexpr int D = 3584, FF = 18944, H = 28, KV = 4, HD = 128;
  int gemm(std::string name, std::string category, int m, int k, int n, std::vector<int> deps) {
    std::vector<Batch> jobs;
    for (auto [mr, mc]: chunks(m, SIDE)) for (auto [nr, nc]: chunks(n, SIDE))
      jobs.push_back({mc * nc, o.sa(mr, k, nr)});
    return g.add(name, category, SA, jobs, deps);
  }
  int ew(std::string name, std::string category, U elements, int cost, std::vector<int> deps) {
    std::vector<Batch> jobs;
    if (elements / EW_CHUNK) jobs.push_back({elements / EW_CHUNK, o.vec(1, EW_CHUNK, cost)});
    if (elements % EW_CHUNK) jobs.push_back({1, o.vec(1, elements % EW_CHUNK, cost)});
    return g.add(name, category, VPU, jobs, deps);
  }
  int rms(std::string name, int rows, std::vector<int> deps) {
    std::vector<Batch> jobs;
    for (auto [r,c]: chunks(rows, ROW_CHUNK)) jobs.push_back({c, o.rms(D, r, nonlinear)});
    return g.add(name, "RMSNorm", VPU, jobs, deps);
  }
  Graph build() {
    int tokens = batch * seq;
    int n1 = rms("norm1", tokens, {});
    int q = gemm("q", "QKV", tokens, D, D, {n1});
    int k = gemm("k", "QKV", tokens, D, KV*HD, {n1});
    int v = gemm("v", "QKV", tokens, D, KV*HD, {n1});
    int qb = ew("q_bias", "QKV_bias", U(tokens)*D, 1, {q});
    int kb = ew("k_bias", "QKV_bias", U(tokens)*KV*HD, 1, {k});
    int vb = ew("v_bias", "QKV_bias", U(tokens)*KV*HD, 1, {v});
    int qr = ew("q_rope", "RoPE", U(tokens)*D, 3, {qb});
    int kr = ew("k_rope", "RoPE", U(tokens)*KV*HD, 3, {kb});
    std::vector<int> av_done;
    for (int b = 0; b < batch; ++b) for (int h = 0; h < H; ++h)
    for (int r = 0; r < seq; r += ROW_CHUNK) {
      int rows = std::min(ROW_CHUNK, seq - r);
      std::string suffix = std::to_string(b)+"_"+std::to_string(h)+"_"+std::to_string(r);
      int sc = gemm("score_"+suffix, "Attention_QK", rows, HD, seq, {qr,kr});
      int sm = g.add("softmax_"+suffix, "Softmax", VPU, {{1, o.softmax(seq, rows, nonlinear)}}, {sc});
      int av = gemm("av_"+suffix, "Attention_AV", rows, seq, HD, {sm,vb});
      av_done.push_back(av);
    }
    int proj = gemm("o", "O_projection", tokens, D, D, av_done);
    int res1 = ew("residual1", "Residual", U(tokens)*D, 1, {proj});
    int n2 = rms("norm2", tokens, {res1});
    int gate = gemm("gate", "Gate_up", tokens, D, FF, {n2});
    int up = gemm("up", "Gate_up", tokens, D, FF, {n2});
    // SiLU(x)*up: negate, exp, add 1, reciprocal, multiply x, multiply up.
    int act = ew("silu_mul", "SiLU_mul", U(tokens)*FF, 4+2*nonlinear, {gate,up});
    int down = gemm("down", "Down_projection", tokens, FF, D, {act});
    ew("residual2", "Residual", U(tokens)*D, 1, {down,res1});
    U expected = U(tokens)*(U(2)*D*D + U(2)*D*KV*HD + U(3)*D*FF)
               + U(2)*batch*H*seq*seq*HD;
    require(g.work()[SA] == expected, "Qwen layer MAC census failed");
    U vector_expected = U(2)*tokens*(4*D+2+nonlinear)
                      + U(tokens)*(D+2*KV*HD)
                      + U(3)*tokens*(D+KV*HD)
                      + U(batch)*H*seq*((6+nonlinear)*seq+nonlinear)
                      + U(2)*tokens*D + U(tokens)*FF*(4+2*nonlinear);
    require(g.work()[VPU] == vector_expected, "Qwen vector census failed");
    return std::move(g);
  }
};

struct Event { U time, serial; int stage, type; };
struct Later { bool operator()(const Event &a, const Event &b) const {
  return std::tie(a.time,a.serial) > std::tie(b.time,b.serial);
}};
struct Bin { U duration=0; std::array<long double,2> busy{}, queued{}; };
struct Category { std::array<U,2> busy{}, work{}; U first=UINT64_MAX,last=0; };
struct Result {
  U cycles=0;
  std::array<U,2> busy{}, work{}, jobs{}, queue_peak{}, queued_cycles{};
  std::map<std::string,Category> categories;
  std::map<U,Bin> bins;
};
Result simulate(const Graph &g, int nsa, int nvu, U bin_width=100000, bool tick_reference=false) {
  std::array<int,2> capacity{nsa,nvu}, free=capacity;
  require(nsa > 0 && nvu > 0, "unit counts must be positive");
  std::array<std::deque<int>,2> ready;
  std::array<U,2> pending{};
  std::vector<int> deps, next_batch(g.stages.size(),0);
  std::vector<U> left, in_batch;
  for (auto &s:g.stages) { deps.push_back(s.deps.size()); left.push_back(s.count()); in_batch.push_back(s.batches[0].count); }
  Result out;
  auto enqueue = [&](int id) {
    int t=g.stages[id].type;
    ready[t].push_back(id);
    pending[t]+=g.stages[id].count();
    out.queue_peak[t]=std::max(out.queue_peak[t],pending[t]);
  };
  for (int i=0;i<(int)g.stages.size();++i) if (!deps[i]) enqueue(i);
  std::priority_queue<Event,std::vector<Event>,Later> events;
  U now=0, serial=0, completed=0;
  auto integrate = [&](U end) {
    U at=now;
    while (at<end) {
      U bi=at/bin_width;
      U stop=std::min(end,(bi+1)*bin_width);
      U dt=stop-at;
      auto &b=out.bins[bi]; b.duration+=dt;
      for (int t=0;t<2;++t) {
        b.busy[t]+=(long double)dt*(capacity[t]-free[t]);
        b.queued[t]+=(long double)dt*pending[t];
        if(pending[t]) out.queued_cycles[t]+=dt;
      }
      at=stop;
    }
  };
  while(completed<g.stages.size()) {
    for(int t=0;t<2;++t) while(free[t]>0 && !ready[t].empty()) {
      int id=ready[t].front(); auto &s=g.stages[id];
      auto &b=s.batches[next_batch[id]];
      --free[t]; --pending[t];
      ++out.jobs[t]; out.busy[t]+=b.timing.cycles; out.work[t]+=b.timing.work;
      auto &cat=out.categories[s.category];
      cat.first=std::min(cat.first,now); cat.busy[t]+=b.timing.cycles; cat.work[t]+=b.timing.work;
      events.push({now+b.timing.cycles,serial++,id,t});
      if(--in_batch[id]==0) {
        ++next_batch[id];
        if(next_batch[id]==(int)s.batches.size()) ready[t].pop_front();
        else in_batch[id]=s.batches[next_batch[id]].count;
      }
    }
    require(!events.empty(),"unfinished graph has no runnable work");
    U until = tick_reference ? now+1 : events.top().time;
    integrate(until); now=until;
    while(!events.empty() && events.top().time==now) {
      auto e=events.top(); events.pop(); ++free[e.type];
      auto &s=g.stages[e.stage];
      out.categories[s.category].last=std::max(out.categories[s.category].last,now);
      if(--left[e.stage]==0) {
        ++completed;
        for(int child:s.children) if(--deps[child]==0) enqueue(child);
      }
    }
  }
  out.cycles=now;
  require(events.empty() && pending==std::array<U,2>{0,0},"work remains after simulation");
  require(out.work==g.work() && out.busy==g.busy(),"simulation failed work/service conservation");
  for(int t=0;t<2;++t) require(out.busy[t]<=U(capacity[t])*now,"capacity violated");
  return out;
}

void selftest(Oracle &o) {
  auto a=o.sa(32,64,32); require(a.cycles==97 && a.work==65536,"native SA timing regression");
  auto v=o.vec(1,16384,1); require(v.cycles==129 && v.work==16384,"native broadcast timing regression");
  auto r=o.vec(64,128,1,true); require(r.cycles==65 && r.work==8192,"native reduction timing regression");
  Layer tile_check{o,{},1,1,1};
  tile_check.gemm("tail","test",35,7,67,{});
  require(tile_check.g.work()[SA]==U(35)*7*67,"tail tile dropped work");
  Graph hand;
  int x=hand.add("x","a",SA,{{4,{3,1}}});
  hand.add("y","b",VPU,{{3,{2,1}}},{x});
  require(simulate(hand,2,1).cycles==12,"hand-calculated dependency chain failed");
  std::mt19937 rng(42);
  for(int test=0;test<50;++test) {
    Graph g;
    for(int i=0;i<12;++i) {
      std::vector<int> ds;
      for(int p=0;p<i;++p) if(rng()%5==0) ds.push_back(p);
      g.add(std::to_string(i),std::to_string(i%3),rng()%2,
            {{1+rng()%5,{1+rng()%11,1+rng()%17}},{1,{1+rng()%3,7}}},ds);
    }
    int ns=1+rng()%4,nv=1+rng()%4;
    auto fast=simulate(g,ns,nv,10), ref=simulate(g,ns,nv,10,true);
    require(fast.cycles==ref.cycles && fast.busy==ref.busy && fast.work==ref.work && fast.queued_cycles==ref.queued_cycles,
            "event schedule disagrees with cycle stepping");
  }
  std::cout<<"PASS: native timings, tail tiles, hand schedule, and 50 event/cycle schedule comparisons\n";
}

int main(int argc,char **argv) {
  try {
    bytes_per_tx=64; mxu_macs_per_pe=1; vmem_reuse=0; act_share=0;
    int batch=64, seq=1024, nsa=144;
    std::string output="results", vpustr="288,144,72,36,18,9,6,4,3,2,1", costs="1,4";
    bool test=false;
    for(int i=1;i<argc;++i) {
      std::string a=argv[i];
      if(a=="--selftest") {test=true;continue;}
      require(i+1<argc,"missing value for "+a);
      std::string value=argv[++i];
      if(a=="--batch") batch=std::stoi(value);
      else if(a=="--seq") seq=std::stoi(value);
      else if(a=="--sa") nsa=std::stoi(value);
      else if(a=="--vpus") vpustr=value;
      else if(a=="--nonlinear-costs") costs=value;
      else if(a=="--out") output=value;
      else throw std::runtime_error("unknown argument "+a);
    }
    require(batch>0 && seq>0 && U(batch)*seq<=10000000 && nsa>0,"invalid dimensions");
    Oracle oracle;
    if(test) {selftest(oracle);return 0;}
    auto integers=[](std::string input) {
      std::vector<int> out; std::istringstream in(input); std::string s;
      while(std::getline(in,s,',')) {int n=std::stoi(s);require(n>0,"list entries must be positive");out.push_back(n);}
      require(!out.empty(),"empty argument list");return out;
    };
    auto vpus=integers(vpustr), nlcosts=integers(costs);
    std::filesystem::create_directories(output);
    std::ofstream summary(output+"/sweep.csv"), timeline(output+"/timeline.csv"), classes(output+"/operators.csv"), census(output+"/work_census.csv");
    require(summary && timeline && classes && census,"could not open result files");
    summary<<"batch,seq,sa_count,vpu_count,sa_side,vpu_lanes,nonlinear_cost,mac_per_vector_op,cycles,time_ms,sa_occupied_pct,vpu_occupied_pct,sa_mac_util_pct,vpu_equiv_util_pct,sa_macs,vpu_equiv_ops,sa_jobs,vpu_jobs,sa_queue_peak,vpu_queue_peak,sa_queued_cycles,vpu_queued_cycles\n";
    timeline<<"nonlinear_cost,vpu_count,start_cycle,duration,sa_occupied_pct,vpu_occupied_pct,sa_queue_mean,vpu_queue_mean\n";
    classes<<"nonlinear_cost,vpu_count,category,first_cycle,last_cycle,sa_busy_cycles,vpu_busy_cycles,sa_macs,vpu_equiv_ops\n";
    census<<"nonlinear_cost,category,sa_macs,vpu_equiv_ops,sa_service_cycles,vpu_service_cycles\n";
    summary<<std::setprecision(12);timeline<<std::setprecision(9);
    for(int nl:nlcosts) {
      Layer model{oracle,{},batch,seq,nl}; Graph graph=model.build();
      bool first=true;
      for(int nv:vpus) {
        auto r=simulate(graph,nsa,nv);
        double ratio=double(nsa)*SIDE*SIDE/(nv*LANES);
        summary<<batch<<','<<seq<<','<<nsa<<','<<nv<<','<<SIDE<<','<<LANES<<','<<nl<<','<<ratio<<','<<r.cycles<<','<<r.cycles/1900000.0
               <<','<<100.0*r.busy[SA]/nsa/r.cycles<<','<<100.0*r.busy[VPU]/nv/r.cycles
               <<','<<100.0*r.work[SA]/nsa/(SIDE*SIDE)/r.cycles<<','<<100.0*r.work[VPU]/nv/LANES/r.cycles
               <<','<<r.work[SA]<<','<<r.work[VPU]<<','<<r.jobs[SA]<<','<<r.jobs[VPU]<<','<<r.queue_peak[SA]<<','<<r.queue_peak[VPU]
               <<','<<r.queued_cycles[SA]<<','<<r.queued_cycles[VPU]<<'\n';
        for(auto &[bin,b]:r.bins) timeline<<nl<<','<<nv<<','<<bin*100000<<','<<b.duration<<','
          <<double(100*b.busy[SA]/nsa/b.duration)<<','<<double(100*b.busy[VPU]/nv/b.duration)<<','
          <<double(b.queued[SA]/b.duration)<<','<<double(b.queued[VPU]/b.duration)<<'\n';
        for(auto &[name,c]:r.categories) {
          classes<<nl<<','<<nv<<','<<name<<','<<c.first<<','<<c.last<<','<<c.busy[SA]<<','<<c.busy[VPU]<<','<<c.work[SA]<<','<<c.work[VPU]<<'\n';
          if(first) census<<nl<<','<<name<<','<<c.work[SA]<<','<<c.work[VPU]<<','<<c.busy[SA]<<','<<c.busy[VPU]<<'\n';
        }
        first=false;
        std::cout<<"nonlinear="<<nl<<" VPU="<<nv<<" MAC:VOP="<<ratio<<":1 cycles="<<r.cycles<<" ms="<<r.cycles/1900000.0<<std::endl;
      }
    }
    std::ofstream timings(output+"/native_timings.csv");timings<<"key,cycles,work\n";
    for(auto &[key,t]:oracle.cache) timings<<key<<','<<t.cycles<<','<<t.work<<'\n';
    std::cout<<"Complete. Native timing shapes: "<<oracle.cache.size()<<std::endl;
  } catch(const std::exception &e) {std::cerr<<"Error: "<<e.what()<<std::endl;return 1;}
}
