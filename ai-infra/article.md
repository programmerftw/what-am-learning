# The Infrastructure Behind AI: From a Single GPU to a Production Inference Cluster

**The short version:** Behind every AI product is a precise infrastructure stack. The core tension is this — a model is a *file*, inference is a *loop*, and serving is a *systems problem*. To understand production-grade AI inference, you need four things: the **two-phase nature of inference (Prefill vs Decode)**, the **KV Cache**, **batching and parallelism strategies**, and **llm-d's cache-aware routing on Kubernetes**.

This year alone, Amazon, Google, Microsoft, and Meta are spending over **$700 billion** on AI infrastructure — four times what they spent four years ago, and up nearly 80% in the last year. McKinsey expects close to **$7 trillion** to go into data centers by 2030. Somebody has to build all of that and keep it running. And almost none of it is machine learning. It's infrastructure work.

---

## Table of Contents

1. [What a Model Actually Is](#1-what-a-model-actually-is)
2. [Why You Can't Just Run It on Your Laptop](#2-why-you-cant-just-run-it-on-your-laptop)
3. [The Model Server: vLLM and Tokens](#3-the-model-server-vllm-and-tokens)
4. [Prefill vs Decode: The Two Halves of the Loop](#4-prefill-vs-decode-the-two-halves-of-the-loop)
5. [The KV Cache and Prefix Caching](#5-the-kv-cache-and-prefix-caching)
6. [Batching: Many Users on One GPU](#6-batching-many-users-on-one-gpu)
7. [Sharding: Splitting Giant Models Across GPUs](#7-sharding-splitting-giant-models-across-gpus)
8. [Why Load Balancing Breaks for LLMs](#8-why-load-balancing-breaks-for-llms)
9. [llm-d: The Smart Router on Kubernetes](#9-llm-d-the-smart-router-on-kubernetes)
10. [The Well-Lit Paths: Pre-Tuned Recipes](#10-the-well-lit-paths-pre-tuned-recipes)
11. [Running llm-d on Kubernetes](#11-running-llm-d-on-kubernetes)
12. [Recap: This Is Infrastructure Work](#12-recap-this-is-infrastructure-work)

---

## 1. What a Model Actually Is

Forget AI for a second. Picture a tiny machine: you drop a number in, a different number comes out. Drop in 3, out comes 6. Drop in 10, out comes 20. It doubles whatever you give it.

That machine is a model. It has two parts:

- **The formula (structure):** input → multiply → output. This part is fixed.
- **The weight (number):** the "2" hiding inside. This part is *learned*.

Change the weight from 2 to 3, and the same formula now triples instead of doubles. Same structure, different number, completely different behavior.

To do something real — say, predict house prices — you need more inputs (size, bedrooms, age) and more weights. Each weight says how much that one thing matters. You don't make those numbers up. You show the formula thousands of real houses with their actual sale prices, and it works out the weights that fit best. **That process is training.**

Language is far more tangled than house prices, so the formula grows: more inputs, more multiplications, more weights, stacked in layers where each layer's output feeds the next. Keep growing it in one very particular arrangement and you arrive at the shape behind every modern language model — the **Transformer**.

> **[DIAGRAM 1 — "The Model as a Machine"]**
> *What to grab from the video:* the simple "input → ×2 → output" box, then the expanded house-price version showing multiple inputs each multiplied by its own weight and summed.
> *Why it matters for study:* it separates "formula" (fixed structure) from "weights" (learned numbers) visually — the single most important mental model in the whole course.

**The key insight:** The Transformer is the *formula*. It's written once in code and is broadly the same across GPT, Claude, and Llama. What makes each model different is the **weights**.

And here's the surprising part: **a model is just a file on disk.** The formula is a page of code; all the knowledge is in the weights. A real model doesn't have four weights or 400 — it has *billions*. They sit as a very large file full of numbers, waiting to be poured into the formula.

| Model size | Parameters | File size on disk |
|---|---|---|
| Small | — | ~2 GB |
| Mid-size | — | ~16 GB |
| Large | 70B | ~140 GB |
| Giant | 500B+ | Several hundred GB |

To use one: load the file into memory, hand it your question, let the formula run all those numbers over your input, get text back. **Load file → fit text → get text.** That's a model running, at its simplest.

---

## 2. Why You Can't Just Run It on Your Laptop

Take a small model that fits in your laptop's memory just fine. Why can't you just run it?

Look at the computer. Two parts matter: the **CPU** (does the thinking, the calculations) and the **RAM** (holds the data the CPU is working on). These two sit *separately*, connected by a channel — a bus — between them.

Every single calculation is a round trip: numbers land in memory, the CPU reads them out, does the math, writes the result back. The CPU is a brilliant **generalist** — a handful of very powerful cores, working through complicated tasks one after another. Fast, and perfect for browsers, apps, and operating systems.

But running a model comes down to **billions of very small multiplications** — and these multiplications *don't depend on each other*. They can all happen at the same time, in parallel. Hand a CPU billions of independent little multiplications and it works through them mostly in sequence, a few at a time. It gets every one right, and you'll be waiting a very, very long time.

**Problem #1: the work is massively parallel; the CPU is built to do things in sequence.**

### GPU Cores, VRAM, and Bandwidth

The GPU is built the exact opposite way. Instead of a handful of powerful cores, it has **thousands of small, simple cores** that all do math at the same time. A modest Nvidia T4 has ~2,500 of these cores.

- CPU: roughly **10 trillion operations/sec**
- GPU: closer to **1,000 trillion operations/sec**

About **100× more math every second.** The GPU solves the math problem — but creates a new one. A processor needs memory to feed it data. A GPU needs something to feed all those cores.

If you leave the weights in the computer's regular RAM, there's a problem: system RAM sits far off the card, down a narrow pipe carrying about **64 Gbps** — roughly **50× slower** than what the cores can eat. Fine for a CPU doing a few things at a time. Useless for thousands of GPU cores all demanding data at once. The cores sit idle, waiting for numbers that arrive too slowly.

The answer: give the GPU its **own memory, built right onto the card**, sitting right next to its cores. This is **VRAM**. Same idea as CPU + RAM, but packed tightly together on a single board so data never travels far. The pipe is measured in **terabytes per second** instead of gigabytes.

But this solution has a catch: **VRAM is fast but small.** A T4 has only 16 GB. A 70B model needs ~140 GB of weights just to sit in memory. Space in VRAM is precious — remember this, because later we'll see how GPUs tackle models hundreds of GB in size across *multiple* GPUs.

> **[DIAGRAM 2 — "CPU + RAM vs GPU + VRAM"]**
> *What to grab from the video:* the side-by-side showing the CPU and RAM as two separate boxes connected by a "road," versus the GPU with its memory packed tightly beside it on one board.
> *Why it matters:* it's the visual explanation for why bandwidth, not raw compute, becomes the bottleneck in Decode.

### The Three Numbers That Size Every GPU

Every card you'll ever meet is described by three numbers:

| Metric | Meaning |
|---|---|
| **Compute** | How much math the card can do |
| **Capacity** | How much fits in its memory |
| **Bandwidth** | How fast it can read its own memory |

Each generation adds more of all three:

| Card | Compute | VRAM | Bandwidth |
|---|---|---|---|
| Desktop | 1–3 TFLOPs | 32–64 GB | ~0.09 TB/s |
| Rack server | 5–10 TFLOPs | high RAM | ~0.5 TB/s |
| **A100** | 312 TFLOPs | 80 GB | ~2 TB/s |
| **H100** | 990 TFLOPs | 80 GB | ~3 TB/s |
| **H200** | 990 TFLOPs | 141 GB | ~4.8 TB/s |
| **B200** | 2,250 TFLOPs | 192 GB | ~8 TB/s |

> **[DIAGRAM 3 — "The GPU Comparison Table"]**
> *What to grab from the video:* the on-screen table comparing desktop, rack server, A100, H100, H200, and B200 across compute, capacity, and bandwidth.
> *Why it matters:* this table is the reference you'll come back to every time you reason about whether a model fits and how fast it can run.

A chip on its own does nothing — something has to drive it. That something is **PyTorch**. In about six or seven lines of code, you import `torch`, load a model, and call a generate command that sends a request and gets a response back. You can try this on your laptop with a really small model.

But that's just a script. It runs once for one person and it's done. Real users are out on the internet — thousands of them, all at once. You need something that's always on, listening for requests, running the model efficiently. **That's a model server.**

---

## 3. The Model Server: vLLM and Tokens

**vLLM** is built right on top of PyTorch. It wraps the model into a service with a normal web API. One command — `vllm serve` — and it reads the model weights off disk, loads them into GPU memory (about 16 GB to move, so you wait a minute or two), and stays up waiting for requests.

Crucially, **vLLM speaks the exact same format as the OpenAI API**. Any tool or script already built for OpenAI can talk to your own server without changing a single line of code.

Notice what you've got: one server holding the entire model in GPU memory the whole time it's running. If one server isn't enough, a second one is *not cheap* — it means another full copy of the model on another expensive GPU. **A model server is a heavy beast: expensive to run, slow to start, costly to copy.**

### Tokens and the One-Token-at-a-Time Loop

Language models don't read and write whole words. They break text into **tokens** — roughly 3/4 of a word. "Serving LLMs is not like serving web apps" is 8 words but about 9 tokens. Everything is counted in tokens: what a model can read, what you pay, how fast it generates.

The model has exactly one trick: **predict the next token, given all the text so far.** Then it appends that token to the text and does it again. And again. A 100-token answer isn't one big calculation — it's the model running that loop 100 times, one token per lap.

That's exactly why the answer types itself out word by word. You're watching each lap finish. Streaming comes as a free feature.

Two consequences fall out of this:

1. **Requests take a long time.** A normal web request finishes in milliseconds; a few hundred tokens means a few hundred laps.
2. **No two requests are the same size.** "Capital of France" is a few tokens in, one or two out. "Summarize this 100-page contract" is tens of thousands of tokens. Both hit the same server through the same API — one is a thousand times bigger than the other.

> **[DIAGRAM 4 — "The Token Loop"]**
> *What to grab from the video:* the circular diagram showing "predict next token → append → predict next token," with the growing text sequence.
> *Why it matters:* everything downstream — latency, cost, batching, caching — is a consequence of this loop.

---

## 4. Prefill vs Decode: The Two Halves of the Loop

Paste a big chunk of text into ChatGPT and hit enter. There's a pause — a second or two where nothing happens — then the answer streams out word by word. **The pause and the stream are the two halves.**

### Prefill (The Pause)

Before the model can write a single word, it has to read everything you gave it — your whole prompt — *at once*. Every token goes through the whole model together in one big burst. The whole model gets loaded out of VRAM once, and thousands of cores fire simultaneously to chew through the entire prompt in a single pass. That pass ends by producing the **first token** of your answer.

- "Capital of France" ≈ 7 tokens → one tiny pass, no noticeable pause.
- A 100-page contract ≈ tens of thousands of tokens → pushed through at once, and that single burst becomes a real wait.

This phase is called **Prefill**. The wait it causes is called **TTFT — Time To First Token**. It is **compute-heavy**.

### Decode (The Stream)

This is the token-by-token loop. Each token is one full pass through the model. To produce each new token, the GPU has to read the **entire model** out of its memory. Writing a 200-token answer means loading the entire model out of memory **200 times**, and each load buys you exactly one token. The cores barely do any work in this phase.

This phase is called **Decode**. Its speed is limited by how fast the GPU can read its own memory. The gap between one token and the next is called **TPOT — Time Per Output Token**.

Quick math: a high-end GPU reads its own memory at ~3 TB/s. Against a 16 GB model, that's the GPU reading the whole thing about **200 times per second** — i.e., ~200 tokens/second max, or one token every ~5 ms. That 5 ms gap is the TPOT.

> **[DIAGRAM 5 — "Prefill vs Decode"]**
> *What to grab from the video:* the two-phase diagram — the initial burst labeled "Prefill / compute-heavy / TTFT" and the steady stream labeled "Decode / memory-bandwidth-heavy / TPOT."
> *Why it matters:* every serving benchmark you'll ever read is built on these two metrics. And the fact that they have *opposite* bottlenecks is what makes the whole rest of the course necessary.

**Summary of the two phases:**

| Phase | What happens | Bottleneck | Metric |
|---|---|---|---|
| **Prefill** | Whole prompt processed in one burst | **Compute** | TTFT |
| **Decode** | Answer written one token at a time | **Memory bandwidth** | TPOT |

One is a one-time calculation. The other is a long, steady grind against memory. Right now, both happen on the same GPU — so the same GPU has to take turns between two completely different kinds of operations.

---

## 5. The KV Cache and Prefix Caching

If every new token required a fresh Prefill of everything before it, chat would be unusably slow — a fresh pause before every single word. That's not what happens.

Prefill is the expensive part, but instead of throwing it away, the model **saves it right there in GPU memory**. That saved work is the **KV Cache**.

Prefill runs once at the start and saves its output. From then on, as the model writes the answer one token at a time, it doesn't repeat Prefill — it reaches into the cache, reuses everything it already worked out, and adds the one new token on top. **You pay for the pause once, not before every word.**

### The Problem Between Messages

Everything above covers exactly *one* message: one question in, one answer out. When the reply finishes, the request is done and the cache gets cleared. So what happens when you send a *second* message in the chat?

The model remembers nothing between messages. Your chat app **resends the entire thread** every time. And because last time's work is already gone, the server has to re-read every word from the very beginning before it can reply. Every turn rebuilds all that work from scratch and throws it away. **The longer you talk, the longer you wait.**

### Prefix Caching: Keep the Cache

What if, instead of throwing the cache away when the request ends, you *kept* it — hanging on to those saved blocks and labeling each one by the exact text that produced it?

Now turn two arrives. The model looks at your thread and realizes it has already done the work on all of it *except your newest message*. That's the only part it actually has to Prefill. Turn 10 now costs about what turn 2 cost. **The pause stays small no matter how long the conversation gets.**

This idea is called **Prefix Caching**. In the Anthropic Console it shows up as "prompt caching," where a cached input token costs about **one-tenth** of a fresh one.

And something else falls out of it for free: the saved work for any stretch of text depends *only on the text before it* — it doesn't know or care whose conversation it's from. If every conversation with your company's assistant opens with the same long block of instructions, the model does that work **once** and reuses it for every person who talks to it.

> **[DIAGRAM 6 — "KV Cache and Prefix Caching"]**
> *What to grab from the video:* the diagram showing a conversation thread where turn 2's cache blocks are reused from turn 1, with only the newest message requiring fresh Prefill.
> *Why it matters:* it's the direct cause of the cost difference between cached and uncached input tokens — and the foundation for understanding llm-d's routing later.

---

## 6. Batching: Many Users on One GPU

Think back to the expensive part of Decode: for every single token, the GPU reads the entire model out of memory. If the GPU already has to read the whole model just to produce one token for one user, **why not use that same read to produce the next token for many users at the same time?**

One read of the model gives you one token for one user, or 50 tokens for 50 users — it's the same single read either way. This is **batching**: pack many people's requests together and run them through the GPU as one group. Each person's own answer still comes out at roughly the same speed, but the server's total output — its **throughput** — shoots up. You serve 50 people for something close to the cost of serving one.

Without batching, you'd be pulling the entire model out of memory for one person at a time, and nobody could afford to run that hardware.

### The Ceiling: Memory

Why can't we batch a million users? **Memory.**

Every user in the batch needs their own KV Cache "scratch pad," and it has to sit in GPU memory for as long as they're being served. That memory is small to begin with, and most of it is already taken by the model weights — which are fixed and always there. Whatever is left over is the only room for everyone's scratch pads.

Fill that space up and that's it. New users wait in line, or somebody else's scratch pad gets thrown out to make room. **This is exactly why ChatGPT sometimes tells you it's at capacity.**

Notice what *isn't* the problem: it's not the processor. There's plenty of math left in those cores, sitting idle. **It's the memory that runs out, and memory decides how many people one GPU can serve.**

This is also where a lot of money gets wasted in the real world: teams buy far more GPUs than the math calls for, and those expensive cards sit half idle.

> **[DIAGRAM 7 — "Batching and the Memory Ceiling"]**
> *What to grab from the video:* the diagram showing multiple users' KV Cache blocks filling up the remaining GPU memory after the model weights take their fixed share, with new users queued at the edge.
> *Why it matters:* it reframes the scaling problem — it's not compute-bound, it's memory-bound — and sets up both sharding and llm-d's scheduling logic.

---

## 7. Sharding: Splitting Giant Models Across GPUs

So far every server holds the whole model. For the biggest models, that doesn't work — a 70B model is ~140 GB, and giants are several hundred GB, while a single GPU may hold only ~80 GB.

The fix: **slice the giant across several GPUs.** No single GPU holds the whole thing, but together they hold the entire model. vLLM can do this on its own — point it at a giant model, tell it how many GPUs it has, and it handles the rest.

A model is a stack of layers. Your prompt enters at the top and flows down, one layer at a time, each layer adding its numbers and passing the result down, until the last one produces the token. There are two ways to split:

**By layer (Pipeline Parallelism):** Hand the first whole layers to GPU 1 and the next ones to GPU 2. Each machine passes just one small result to the next — **light chatter.**

**Within each layer (Tensor Parallelism):** Take a single layer's sum and split its terms. GPU 1 adds the first half, GPU 2 adds the second, and the two halves combine into that layer's answer. This requires constant, **chatty** communication.

Which one you use comes down to **the speed of the wire between the GPUs:**

- **Inside a single machine:** GPUs are joined by a special ultra-fast link called **NVLink**, far quicker than any ordinary network. The chatty method (Tensor Parallel) works well here. Pack eight GPUs into one box on that fast link and they serve as if they were one machine. **This is how most large models run today.**
- **Across machines:** GPUs only have the ordinary network between them, which is much slower. So you switch to the light method (Pipeline Parallel) — hand out whole stretches of layers, and each machine passes one small result on.

**The rule: keep the chatty talk inside a box, and the light talk between boxes.**

This technique is called **Sharding**. It solves the *fitting* problem — a giant too big for one GPU now runs across a machine's worth of GPUs, or several machines' worth acting as one logical server.

> **[DIAGRAM 8 — "Pipeline vs Tensor Parallelism"]**
> *What to grab from the video:* the side-by-side showing layers split across GPUs (pipeline) versus a single layer's sum split across GPUs (tensor), with NVLink inside a box and slower network between boxes.
> *Why it matters:* it's the physical-layout rule that determines which parallelism strategy you can use — and it comes back when llm-d spans pools across machines.

---

## 8. Why Load Balancing Breaks for LLMs

A real service is never run on a single server — you need a whole fleet, each carrying its own copy of the model. The obvious move is a load balancer in front, spreading requests round-robin. We've balanced web traffic this way for decades.

For LLM serving, it's the wrong move, for two reasons.

**Reason 1: Your saved work is stuck on one server.** Say your first message lands on Server 2, which reads your conversation and saves its work in its KV Cache. A little later you send a second message, but the load balancer's whole job is to spread traffic evenly — so it sends this one to Server 5. Server 5 has never seen you. It has nothing saved and has to read your entire conversation again from scratch. **The load balancer treats every server as interchangeable, but we already know they're not.** It throws away perfectly good saved work — on the most expensive hardware you own.

**Reason 2: It assumes all requests are roughly the same size.** Spreading them evenly is supposed to spread the load evenly. But no two LLM requests are the same size. One might be a quick "hello"; the next is "summarize 50 pages of text." To the load balancer, these look identical — same address, same kind of traffic. So it might send that 50-page request to a server already streaming answers to 30 other people, and all 30 slow down.

The load balancer can't tell any of this apart because everything it would need to know is *inside* the servers, where it has no way of seeing it: How full is each server's memory right now? Whose saved work sits where? How long is each queue?

**Serving LLMs needs a much smarter way to route requests than anything we've used before.**

> **[DIAGRAM 9 — "Why Round-Robin Fails"]**
> *What to grab from the video:* the diagram showing a request landing on Server 2, its KV Cache being saved there, then the next request being round-robined to Server 5 which has no cache — forcing a full re-Prefill.
> *Why it matters:* it's the concrete failure that motivates llm-d. If you can explain this diagram, you understand the entire routing problem.

### The Wall Everyone Hit

To recap why serving models is so hard:

1. The model barely fits in memory to begin with.
2. Every token forces the server to re-read the whole thing.
3. Whatever saved work a server has done for you is stuck on that one server.
4. Memory caps how many people that server can handle at once.
5. Get the routing wrong and you waste your most expensive hardware — GPU cycles.

That wall stopped everyone in the industry. So Red Hat, Google, IBM, and Nvidia decided to solve it together, in the open. What they built is **llm-d**.

---

## 9. llm-d: The Smart Router on Kubernetes

At its heart, llm-d is a **smart router** sitting in front of your entire fleet. When a request comes in, it doesn't just throw it at whichever server happens to be free. It **stops and thinks** about where that request should go, then sends it to the server best equipped to handle it.

To make that call, it looks at three things:

### 1. Cache-Aware Routing (the saved work)

llm-d keeps track of which server holds which saved work. When your next message comes in, instead of scattering it across the fleet, llm-d sends it **straight back to the server that already has your conversation.** That server skips the re-read, and your reply comes back instantly.

Routing this way instead of spreading evenly gives you around **3× the throughput** and a first response that is **twice as fast** on the same hardware.

### 2. Load-Aware Routing (the server's state)

llm-d looks inside each server at details a plain load balancer never checks: How full is its memory right now? How long is its queue? It uses that to send your request to a server that genuinely has room, instead of piling work into one that's already full.

### 3. Prefill/Decode Disaggregation (the two halves)

A request has two very different halves. **Prefill** is compute-heavy — the one-time job of reading a prompt. **Decode** is memory-heavy — the slow token-by-token job of writing the answer. On one shared GPU, those two take turns. So when someone pastes in 50 pages, that one big Prefill makes everyone's answer wait mid-stream.

llm-d can **split them onto separate pools**: one pool does nothing but Prefill, the other does nothing but Decode. Prefill is all compute; Decode is all memory — so each pool runs the hardware that fits its job. For example, H100-class GPUs for the Prefill pool, and H200-class (higher memory) GPUs for the Decode pool. When a Prefill server finishes reading a prompt, it hands the saved work across a fast link to a Decode server, which streams the response.

That's up to **70% more tokens per second on the same hardware.**

> **[DIAGRAM 10 — "llm-d Architecture"]**
> *What to grab from the video:* the fleet diagram showing the Gateway and llm-d scheduler out front, with separate Prefill and Decode pools behind, and the KV Cache handoff arrow between them.
> *Why it matters:* it's the single most important architecture diagram in the course. Everything else is a detail of this picture.

### The Platform Underneath: Kubernetes

The best part: llm-d runs on top of something you're almost certainly already using. It's the platform 60% of organizations hosting GenAI use to manage inference workloads. It's what your ops team already runs, and it's becoming the de facto standard for AI inference.

**That platform is Kubernetes.**

A recent CNCF blog ("The Future of AI is Community-Driven and Open") notes that **66% of organizations hosting generative AI** use Kubernetes to manage some or all of their inference workloads. About two out of three teams running generative AI in production say Kubernetes — and llm-d sits right on top of it.

---

## 10. The Well-Lit Paths: Pre-Tuned Recipes

Everything we've talked about has a real, **tunable knob** behind it:

- How much should the router favor a server that already holds your saved work over one that's simply less busy?
- Should you split the servers that read prompts from the ones that write answers?
- How many of each do you run?
- How many GPUs does one copy of the model span when it's too big for a single machine?
- How many requests does each server batch before it starts falling behind?

Every one of these moves your **speed** and your **cost**. And the right answer depends on your model, your hardware, and your traffic. Tune them all by hand and you could spend weeks and still get several wrong.

So llm-d hands you what it calls the **Well-Lit Paths**: ready-made recipes where every knob is already set, each one measured on real hardware. Three carry almost the whole story:

**Path 1 — Optimized Baseline.** One pool of plain, identical servers with cache-aware routing switched on, nothing else changed. This is the smallest change you can make and the biggest payoff. **Most people start here.**

**Path 2 — Prefill/Decode Disaggregation.** Two pools: one reads prompts (Prefill), one writes answers (Decode), with the saved work handed between them. Size each pool for your own traffic. This is the path for **long, heavy prompts**.

**Path 3 — Multi-GPU Sharding.** A single model spread across a group of GPUs that act as one server. This is what the **giants** need.

The router stays exactly the same across all three. The only thing that changes is the shape of the fleet behind it.


---

## 11. Running llm-d on Kubernetes

Setting it all up takes **two commands**:

```bash
helm install router ...   # installs the llm-d router and sets up the vLLM servers
helm install model ...    # installs the model itself
```

Here's what they build:

1. **Start with a Kubernetes cluster** with some worker nodes — nothing running on them yet.
2. **Model servers come first.** Every one is a vLLM instance — the same server from earlier. On Kubernetes, each vLLM instance runs as a **Pod**, landing on whichever node has room. A group of identical pods is a **Deployment**.
3. **Two pools:** your Prefill pool and your Decode pool — a few different pods on a few different servers.
4. **Out front, a Gateway** — the same kind of gateway you'd put in front of any web service. It gives you **one address for the whole fleet**.
5. **Right behind it, the llm-d scheduler** — the small brain holding the cache-aware and load-aware logic. The Gateway takes the request; the scheduler picks the pod.

### StatefulSets vs LeaderWorkerSet

Say a model is too big for a single machine — it has to span several. Kubernetes needs a way to treat that whole group of pods as **one unit**.

If you know Kubernetes, you might think **StatefulSet**. It gives each part a stable name and starts them in a fixed order (0, 1, 2, 3...), which is great for databases like MySQL that need to power up in sequence. **But it still treats every pod as its own separate replica.**

That doesn't fit LLMs split across pods. They're **not separate servers or replicas** — together they are *one server*, simply split into pieces. Scaling should add **another whole group**, not a single pod that would just hang there. And if a single part fails, the whole group has to restart together, because **a model missing a piece can't answer anything at all.**

So Kubernetes built a new kind of object: the **LeaderWorkerSet**. It's a Leader pod plus its Workers, scaled and healed as **a single unit**.

### Describing It: values.yaml

Every box in the architecture is a standard Kubernetes object. You describe it all through the `values.yaml` file you pass to Helm:

- The URI to the model — where the model file lives
- How many pods serve **Prefill**
- How many pods serve **Decode**

Swap that one file for another, and you get a completely different fleet of servers and pods. Apply it, and llm-d creates the Prefill pods and Decode pods, puts the Gateway and scheduler out front, and wires it all together. A minute later, you ask the cluster what's running and it tells you exactly what's in your inference system.

### Keeping It Running

This is the best part: because it's deployed on Kubernetes, **every server here is just a pod.** The normal Kubernetes loop looks after all of them — the part your ops team already trusts:

- A pod crashes → Kubernetes restarts it.
- A whole machine dies → Kubernetes moves those pods somewhere else.
- Readers and writers are separate pools → grow each independently. Add Decode pods when answers back up; add Prefill pods when prompts get heavy.

**Kubernetes keeps these servers alive; llm-d keeps them smart.**

