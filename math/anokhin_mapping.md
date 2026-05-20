# Mapping: P.K. Anokhin's Theory of Functional Systems ↔ GRA Health Swarm

## 1. Core Concepts

| Anokhin's Concept | GRA-Health-Swarm Equivalent | Description |
|-------------------|----------------------------|-------------|
| **Functional System (ФС)** | Swarm of health agents | Dynamic organization of elements to achieve a useful adaptive result |
| **Useful Adaptive Result** | Φ_total → 0 (coherent health state) | The system-forming factor — what the system is built to achieve |
| **Afferent Synthesis** | `AfferentSynthesisAgent` | Integration of external stimuli, internal state, and memory |
| **Decision-making** | `DecisionDominanceAgent` | Formation of dominant goal |
| **Acceptor of Action Result (АРД)** | `AcceptorOfActionResultAgent.prediction` | Predictive model of future result |
| **Efferent Action** | `EfferentActionAgent` | Motor/physiological execution |
| **Reverse Afferentation** | `ReverseAfferentationAgent` | Feedback about action results |
| **Mismatch (Рассогласование)** | Φ (foam) | Difference between prediction and reality |
| **Orienting-Exploratory Reaction** | N_health nullification step | Response to mismatch → recalibration |
| **Chronic Mismatch → Disease** | Chronically high Φ_total | Persistent prediction-reality gap |
| **Compensatory Systems** | Backup AAR agents | Alternative functional systems for recovery |
| **Systemogenesis** | Rejuvenation loop | Developmental/healing formation of new functional systems |

## 2. Mathematical Mapping

### Anokhin's AAR mechanism:
AAR(t) = f(goal, past_experience, current_context)
Result(t) = Action(AAR(t)) + noise
Mismatch(t) = |Result(t) - AAR(t)|
If Mismatch(t) > threshold → stress → recalibration

text

### GRA-Health-Swarm formalization:
Ψᵢ(t) = state of agent i at time t
Φ(Ψᵢ) = internal foam of agent i
Φ_cross(Ψᵢ, Ψⱼ) = mismatch between agents i and j
Φ_total = Σᵢ Φ(Ψᵢ) + Σᵢⱼ Φ_cross(Ψᵢ, Ψⱼ)

N_health: (Ψ₁, ..., Ψₙ) → (Ψ'₁, ..., Ψ'ₙ)
such that Φ_total(Ψ') < Φ_total(Ψ)

Health ≡ Φ_total ≈ 0
Disease ≡ Φ_total ≫ 0 (chronic)
Rejuvenation ≡ Φ_total monotonically decreasing → 0

text

## 3. Key Insight

Anokhin's functional system IS a swarm — a temporary coalition of diverse anatomical elements united by a common goal (useful adaptive result). GRA-Health-Swarm formalizes this coalition as a multi-agent system where:

- Each "element" = an agent with state Ψᵢ
- The "common goal" = Φ_total → 0
- The "AAR" = predictive model maintained by specialized agents
- "Healing" = iterative application of N_health to reduce Φ_total

This mapping shows that Anokhin was essentially describing a **biological GRA nullification process** — 50 years before the formal framework existed.