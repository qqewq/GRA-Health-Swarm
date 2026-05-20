"""Tests for rejuvenation loop."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core import RejuvenationLoop


def test_rejuvenation_loop_basic():
    """Test that rejuvenation loop runs without errors."""
    loop = RejuvenationLoop(dim=5, learning_rate=0.1)

    # Run a few steps
    for step in range(50):
        ext = loop.target_state + np.random.normal(0, 0.05, loop.dim)
        int_state = loop.target_state + np.random.normal(0, 0.1, loop.dim)
        metrics = loop.step(ext, int_state, stress=0.1, support=0.8)

    summary = loop.get_current_state_summary()
    assert summary['phi_total'] >= 0
    assert 0 <= summary['health_index'] <= 1
    print(f"After 50 steps: Φ_total = {summary['phi_total']:.4f}, "
          f"Health = {summary['health_index']:.4f}")


def test_rejuvenation_convergence():
    """Test that system can converge to healthy state under ideal conditions."""
    loop = RejuvenationLoop(dim=5, learning_rate=0.2)

    # Set up ideal conditions: high adaptability, low stress
    for agent in loop.agents.values():
        agent.adaptability = 0.9

    metrics = loop.run_until_healthy(
        max_steps=500,
        health_threshold=0.2
    )

    summary = loop.get_current_state_summary()
    print(f"Convergence test: Φ_total = {summary['phi_total']:.4f}")
    print(f"Steps to health: {metrics.total_steps_to_health}")
    print(f"Relapses: {metrics.relapses}")

    # Under ideal conditions, should converge
    assert metrics.total_steps_to_health is not None or summary['phi_total'] < 0.5


def test_stress_impact():
    """Test that high stress prevents convergence."""
    loop_high_stress = RejuvenationLoop(dim=5, learning_rate=0.1)
    loop_low_stress = RejuvenationLoop(dim=5, learning_rate=0.1)

    # Run both with different stress levels
    for step in range(200):
        ext = loop_high_stress.target_state + np.random.normal(0, 0.05, 5)
        int_state = loop_high_stress.target_state + np.random.normal(0, 0.1, 5)
        loop_high_stress.step(ext, int_state, stress=0.9, support=0.2)
        loop_low_stress.step(ext, int_state, stress=0.1, support=0.8)

    phi_high = loop_high_stress.health_state.phi_total
    phi_low = loop_low_stress.health_state.phi_total

    print(f"High stress Φ_total: {phi_high:.4f}")
    print(f"Low stress Φ_total:  {phi_low:.4f}")

    # High stress should result in higher foam
    assert phi_high > phi_low


if __name__ == "__main__":
    test_rejuvenation_loop_basic()
    test_rejuvenation_convergence()
    test_stress_impact()
    print("All rejuvenation tests passed! ✓")
Итоговая инструкция по созданию репозитория
Шаг 1: Создать репозиторий на GitHub
Перейти на https://github.com/qqewq

Нажать New repository

Название: GRA-Health-Swarm

Описание: Swarm Subjectivity for Health, Rejuvenation, Stress Resilience, and Healing — via Anokhin's Functional Systems and GRA Nullification

Публичный

НЕ добавлять README, .gitignore, лицензию (мы создадим свои)

Шаг 2: Клонировать и создать файлы
bash
git clone https://github.com/qqewq/GRA-Health-Swarm.git
cd GRA-Health-Swarm
mkdir -p core scenarios/drone_phobia scenarios/rejuvenation scenarios/burnout scenarios/chronic_disease math notebooks docs/diagrams paper tests
Затем создать все файлы из этого ответа в соответствующих директориях.

Шаг 3: Запушить
bash
git add .
git commit -m "Initial commit: GRA-Health-Swarm — Anokhin + GRA for rejuvenation and healing"
git push origin main
Шаг 4: Запустить демонстрацию
bash
pip install -r requirements.txt
python scenarios/rejuvenation/run.py
Репозиторий готов. Он объединяет:

🧠 Теорию функциональных систем П.К. Анохина

🔄 GRA-обнулёнку (Φ → 0)

🐝 Субъектные рои (заражение субъектностью)

💊 Механизм омоложения через рекалибровку АРД

Если нужно добавить ещё сценарии (burnout, chronic_disease), доработать paper_ru.tex или создать диаграммы — скажите.