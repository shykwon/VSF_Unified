#!/usr/bin/env python
"""
VSF Research Platform - Parallel Experiment Runner

3개 GPU에서 모든 모델을 병렬로 실험하는 스크립트

Usage:
    # 전체 실험 실행
    python scripts/run_parallel.py

    # 특정 데이터셋만
    python scripts/run_parallel.py --dataset metr-la

    # Dry-run (실제 실행 없이 명령어만 확인)
    python scripts/run_parallel.py --dry-run

    # 특정 모델만
    python scripts/run_parallel.py --models fdw,ginar
"""

import argparse
import subprocess
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# ============================================================================
# Configuration
# ============================================================================

# GPU 할당 전략: {gpu_id: [models]}
# 가벼운 모델은 같은 GPU에, 무거운 모델은 단독 GPU
# NOTE: 2 GPU 환경으로 재배치 (GPU 2 사용 불가)
GPU_ASSIGNMENT = {
    0: ['fdw', 'ginar', 'saits', 'gimcc'],  # 가벼운/중간 모델들 (순차 실행)
    1: ['csdi', 'srdi'],                     # Diffusion 모델들 (순차 실행)
}

# 모델별 권장 batch_size (GTX 1080 11GB 기준)
MODEL_BATCH_SIZES = {
    'fdw': 32,
    'ginar': 32,
    'csdi': 8,      # Diffusion: VRAM 많이 사용
    'srdi': 4,      # Diffusion+Dispatcher: OOM 방지를 위해 축소
    'saits': 16,    # Attention: 중간
    'gimcc': 16,    # Graph+Causal: 중간
}

# 기본 설정
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_DATASETS = ['metr-la']  # 필요시 'pems-bay' 추가
DEFAULT_EPOCHS = 100


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(gpu_id, model, dataset, seeds, epochs, log_dir, dry_run=False):
    """단일 실험 실행"""
    batch_size = MODEL_BATCH_SIZES.get(model, 32)
    seeds_str = ','.join(map(str, seeds))

    # NOTE: CUDA_VISIBLE_DEVICES 설정 시 해당 GPU가 cuda:0으로 매핑됨
    cmd = [
        sys.executable, 'scripts/train.py',
        '--model', model,
        '--dataset', dataset,
        '--seeds', seeds_str,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--log_dir', log_dir,
        '--device', 'cuda:0',  # CUDA_VISIBLE_DEVICES로 GPU 선택하므로 항상 cuda:0
        '--tensorboard'
    ]

    cmd_str = ' '.join(cmd)
    env_str = f"CUDA_VISIBLE_DEVICES={gpu_id}"

    print(f"\n{'='*60}")
    print(f"[GPU {gpu_id}] Starting: {model.upper()} on {dataset}")
    print(f"Command: {env_str} {cmd_str}")
    print(f"{'='*60}\n")

    if dry_run:
        return {
            'model': model,
            'dataset': dataset,
            'gpu': gpu_id,
            'status': 'dry-run',
            'command': f"{env_str} {cmd_str}"
        }

    # 실제 실행
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    start_time = time.time()
    try:
        # 출력을 실시간으로 표시
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 로그 파일에도 저장
        log_file = Path(log_dir) / f"{model}_{dataset}_gpu{gpu_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'w') as f:
            for line in process.stdout:
                print(f"[GPU {gpu_id}] {line}", end='')
                f.write(line)

        process.wait()
        elapsed = time.time() - start_time

        return {
            'model': model,
            'dataset': dataset,
            'gpu': gpu_id,
            'status': 'success' if process.returncode == 0 else 'failed',
            'returncode': process.returncode,
            'elapsed_seconds': elapsed,
            'log_file': str(log_file)
        }

    except Exception as e:
        return {
            'model': model,
            'dataset': dataset,
            'gpu': gpu_id,
            'status': 'error',
            'error': str(e)
        }


def run_gpu_experiments(gpu_id, models, datasets, seeds, epochs, log_dir, dry_run=False):
    """특정 GPU에서 할당된 모델들을 순차 실행"""
    results = []

    for model in models:
        for dataset in datasets:
            result = run_experiment(
                gpu_id, model, dataset, seeds, epochs, log_dir, dry_run
            )
            results.append(result)

            if not dry_run and result['status'] == 'success':
                print(f"\n✅ [GPU {gpu_id}] {model.upper()} on {dataset} completed!")
            elif not dry_run:
                print(f"\n❌ [GPU {gpu_id}] {model.upper()} on {dataset} failed!")

    return results


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='VSF Parallel Experiment Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--models', type=str, default=None,
                        help='Models to run (comma-separated). Default: all')
    parser.add_argument('--datasets', type=str, default='metr-la',
                        help='Datasets to run (comma-separated)')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Seeds for experiments (comma-separated)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--log_dir', type=str, default='logs/parallel',
                        help='Log directory')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='GPUs to use (comma-separated)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--sequential', action='store_true',
                        help='Run all experiments sequentially (single GPU)')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt')

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse arguments
    datasets = [d.strip() for d in args.datasets.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    gpus = [int(g.strip()) for g in args.gpus.split(',')]

    if args.models:
        selected_models = [m.strip().lower() for m in args.models.split(',')]
    else:
        selected_models = None  # All models

    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"{args.log_dir}_{timestamp}"

    print(f"\n{'='*60}")
    print("    VSF Parallel Experiment Runner")
    print(f"{'='*60}")
    print(f"  Datasets: {datasets}")
    print(f"  Seeds:    {seeds}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  GPUs:     {gpus}")
    print(f"  Log dir:  {log_dir}")
    print(f"  Dry-run:  {args.dry_run}")
    print(f"{'='*60}\n")

    # Filter GPU assignment based on available GPUs and selected models
    gpu_tasks = {}
    for gpu_id in gpus:
        if gpu_id in GPU_ASSIGNMENT:
            models = GPU_ASSIGNMENT[gpu_id]
            if selected_models:
                models = [m for m in models if m in selected_models]
            if models:
                gpu_tasks[gpu_id] = models

    # Show experiment plan
    print("Experiment Plan:")
    print("-" * 40)
    total_experiments = 0
    for gpu_id, models in gpu_tasks.items():
        for model in models:
            for dataset in datasets:
                print(f"  GPU {gpu_id}: {model.upper():6s} on {dataset}")
                total_experiments += 1
    print("-" * 40)
    print(f"Total: {total_experiments} experiments x {len(seeds)} seeds = {total_experiments * len(seeds)} runs")
    print()

    if args.dry_run:
        print("\n[DRY-RUN MODE] Commands that would be executed:\n")
        for gpu_id, models in gpu_tasks.items():
            for model in models:
                for dataset in datasets:
                    run_experiment(gpu_id, model, dataset, seeds, args.epochs, log_dir, dry_run=True)
        return

    # Confirm before running
    if not args.dry_run and not args.yes:
        response = input("\nProceed with experiments? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Run experiments
    start_time = time.time()
    all_results = []

    if args.sequential:
        # Sequential execution on single GPU
        print("\n[SEQUENTIAL MODE] Running all experiments on single GPU...")
        gpu_id = gpus[0]
        all_models = [m for models in gpu_tasks.values() for m in models]
        results = run_gpu_experiments(
            gpu_id, all_models, datasets, seeds, args.epochs, log_dir
        )
        all_results.extend(results)
    else:
        # Parallel execution across GPUs
        print("\n[PARALLEL MODE] Starting experiments across GPUs...")

        with ProcessPoolExecutor(max_workers=len(gpu_tasks)) as executor:
            futures = {}
            for gpu_id, models in gpu_tasks.items():
                future = executor.submit(
                    run_gpu_experiments,
                    gpu_id, models, datasets, seeds, args.epochs, log_dir
                )
                futures[future] = gpu_id

            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"\n✅ GPU {gpu_id} completed all experiments!")
                except Exception as e:
                    print(f"\n❌ GPU {gpu_id} failed: {e}")

    # Summary
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("    Experiment Summary")
    print(f"{'='*60}")

    success_count = sum(1 for r in all_results if r.get('status') == 'success')
    failed_count = sum(1 for r in all_results if r.get('status') != 'success')

    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {failed_count}")
    print()

    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_time_seconds': total_time,
        'config': vars(args),
        'results': all_results
    }

    summary_path = Path(log_dir) / 'experiment_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")

    # Print individual results
    print("\nResults by model:")
    print("-" * 40)
    for r in all_results:
        status_icon = "✅" if r.get('status') == 'success' else "❌"
        elapsed = r.get('elapsed_seconds', 0)
        print(f"  {status_icon} {r['model'].upper():6s} on {r['dataset']:10s} "
              f"(GPU {r['gpu']}) - {elapsed/60:.1f}min")

    print(f"\n{'='*60}")
    print("    Done!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
