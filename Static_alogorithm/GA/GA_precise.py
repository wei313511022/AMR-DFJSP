import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

from GA import (
    DISPATCH_EVENT_INDEX_ENV,
    GENERATIONS,
    POPULATION_SIZE,
    STAGNATION_LIMIT,
    Individual,
    Job,
    describe_solution,
    fitness,
    greedy_individual,
    load_dispatch_events,
    local_improve,
    make_jobs,
    mutate,
    order_crossover,
    random_individual,
    routing_iters,
)


def get_parent_via_tournament(population_scored, k=3):
    candidates = random.sample(population_scored, min(k, len(population_scored)))
    winner = min(candidates, key=lambda x: x[0])
    return winner[1]


def evolve_precise(
    jobs: List[Job],
    init_state: dict = None,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    local_iters: int = routing_iters,
    verbose: bool = True,
) -> Tuple[Individual, List[Tuple]]:
    """
    GA variant that optimizes directly against the collision-aware evaluator.

    Standard GA.py uses rough fitness during most of evolution and only applies
    collision-aware evaluation/improvement near the end. This precise version
    uses check_collision=True for selection, archive tracking, and local search.
    """
    pop_random_count = int(population_size * 0.8)
    population = [random_individual(jobs) for _ in range(pop_random_count)]
    population += [greedy_individual(jobs) for _ in range(population_size - pop_random_count)]

    archive_best: Individual = population[0]
    best_fitness = float("inf")
    best_timeline: List[Tuple] = []
    stagnation_counter = 0

    for gen in range(generations):
        scored = []
        for ind in population:
            score, _ = fitness(ind, jobs, check_collision=True, init_state=init_state)
            scored.append((score, ind))

        scored.sort(key=lambda pair: pair[0])
        current_best = scored[0][1]
        current_score = scored[0][0]

        if current_score < best_fitness:
            best_fitness = current_score
            best_timeline = fitness(
                current_best,
                jobs,
                check_collision=True,
                init_state=init_state,
            )[1]
            archive_best = Individual(
                order=list(current_best.order),
                amr_assignment=list(current_best.amr_assignment),
            )
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter > STAGNATION_LIMIT:
            elite_count = min(5, population_size)
            population = [
                Individual(order=list(pair[1].order), amr_assignment=list(pair[1].amr_assignment))
                for pair in scored[:elite_count]
            ]
            population += [random_individual(jobs) for _ in range(population_size - elite_count)]
            stagnation_counter = 0
            continue

        new_generation = []
        for _, elite_ind in scored[:2]:
            new_generation.append(
                Individual(order=list(elite_ind.order), amr_assignment=list(elite_ind.amr_assignment))
            )

        while len(new_generation) < population_size:
            parent_a = get_parent_via_tournament(scored, k=3)
            parent_b = get_parent_via_tournament(scored, k=3)
            child = order_crossover(parent_a, parent_b, jobs)
            mutate(child, jobs, init_state=init_state)
            new_generation.append(child)

        population = new_generation

        if verbose and (gen == 0 or (gen + 1) % 10 == 0 or gen + 1 == generations):
            print(f"Generation [{gen + 1}/{generations}] | Best precise fitness: {best_fitness:.2f}")

    if local_iters > 0:
        archive_best = local_improve(
            archive_best,
            jobs,
            max_iters=local_iters,
            check_collision=True,
            init_state=init_state,
        )

    _, best_timeline = fitness(
        archive_best,
        jobs,
        check_collision=True,
        init_state=init_state,
    )
    return archive_best, best_timeline


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    if args.inbox:
        dispatch_events = load_dispatch_events(Path(args.inbox))
    else:
        dispatch_events = load_dispatch_events()

    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)
    if dispatch_events and target_index is not None:
        dispatch_events = [event for event in dispatch_events if str(event["index"]) == str(target_index)]

    results_data = []

    print("=== Using GA Precise Logic (Collision-Aware Fitness During Evolution) ===")

    if dispatch_events:
        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")
            start_time = time.perf_counter()
            best_ind, _ = evolve_precise(
                event["jobs"],
                population_size=args.population,
                generations=args.generations,
                local_iters=args.local_iters,
            )
            solve_dur = time.perf_counter() - start_time

            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            makespan, computation_time = describe_solution(
                best_ind,
                event["jobs"],
                solve_time=solve_dur,
                show_gantt=args.gantt,
                save_img=img_path,
            )
            results_data.append(
                [
                    event["index"],
                    f"{makespan:.2f}",
                    f"{computation_time:.4f}" if computation_time is not None else "0.0000",
                ]
            )
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = make_jobs()
        start_time = time.perf_counter()
        best_ind, _ = evolve_precise(
            jobs,
            population_size=args.population,
            generations=args.generations,
            local_iters=args.local_iters,
        )
        solve_dur = time.perf_counter() - start_time

        makespan, computation_time = describe_solution(
            best_ind,
            jobs,
            solve_time=solve_dur,
            show_gantt=args.gantt,
            save_img=args.save_img,
        )
        results_data.append(
            [
                "random",
                f"{makespan:.2f}",
                f"{computation_time:.4f}" if computation_time is not None else "0.0000",
            ]
        )

    if results_data:
        with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Event_Index", "Makespan", "Computation_Time"])
            writer.writerows(results_data)
        print(f"\nSummary results saved to {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gantt", action="store_true", help="Plot Gantt Chart")
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--save_img", type=str, default="", help="Save the schedule Gantt chart to this file")
    parser.add_argument("--output_csv", type=str, default="GA_precise_summary_results.csv")
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--local_iters", type=int, default=routing_iters)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
