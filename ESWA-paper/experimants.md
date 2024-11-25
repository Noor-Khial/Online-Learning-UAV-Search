This file describes the experiments conducted for the paper, based on the methodologies detailed in:  
**Khial, Noor, Mhaisen, Naram, Mabrok, Mohamed, and Mohamed, Amr.**  
_"An Online Learning Framework for UAV Search Mission in Adversarial Environments."_  
Available at SSRN: [SSRN 4725375](https://ssrn.com/abstract=4725375).

## Section 6.1.1. Impact of Context Scaling on C-Exp3 Performance

- Description: Fine-tune the number of contexts.
- Parameters: Set `|\mathcal{C}| ∈ {9, 16, 36}`.
- Script: Use the optimal `η` parameter in `cexp3.py`.

## Section 6.1.2. Fixed Best Action

- Description: Evaluate performance with a fixed mobility pattern.
- Parameters: Set `|\mathcal{C}| = 36` and use the optimal `η`.
- Script: Run `cexp3.py` with a fixed mobility pattern throughout `\mathcal{T}`.

## Section 6.1.3. Comparison with State-of-the-Art Solutions

- Description: Compare the proposed approach with state-of-the-art methods.
- Methods: Use existing implementations of:
  - DQN from [Deep Q-Learning Repository](https://github.com/keon/deep-q-learning.git).
  - PPO from [Reinforcement Learning PPO Repository](https://github.com/wisnunugroho21/reinforcement_learning_truly_ppo.git).
  - Modified Q-Table implementation.
- Evaluation: Measure performance in terms of average regret as shown in Table 4 of the manuscript.

## Section 6.1.4. Changing Best Action

- Description: Evaluate performance with varying mobility patterns.
- Parameters: Set `|\mathcal{C}| = 36`, use the optimal `η`, and apply:
  - Mobility pattern 1 during `\mathcal{T}_1`.
  - Mobility pattern 2 during `\mathcal{T}_2`.
- Script: Run `cexp3.py`.

## Section 6.2. Performance of the Agent with AC-Exp3 Algorithm

- Description: Similar setup to Section 6.1.4 with action weights reinitialized.
- Parameters: Set `|\mathcal{C}| = 36`, use the optimal `η`, and apply:
  - Mobility pattern 1 during `\mathcal{T}_1`.
  - Mobility pattern 2 during `\mathcal{T}_2`.
- Script: Run `cexp3.py`.

## Section 6.3.1. Changing Best Action

- Description: Evaluate Exp4 algorithm with varying experts and mobility patterns.
- Parameters:
  - Set the number of experts to 2.
  - Set the frequency of mobility pattern changes to 2 during `\mathcal{T}_1` and `\mathcal{T}_2`.
  - Apply mobility patterns:
    - Mobility pattern 1 during `\mathcal{T}_1`.
    - Mobility pattern 2 during `\mathcal{T}_2`.
- Script: Run `exp4.py`.

## Section 6.3.2. Impact of the Number of Experts on Exp4

- Description: Investigate the effect of expert count on performance.
- Parameters:
  - Fine-tune the number of experts to `{1, 3, 5, 7}`.
  - Set the frequency of mobility pattern changes to 6 during:
    - `\mathcal{T}_1, \mathcal{T}_2, \mathcal{T}_3, \mathcal{T}_4, \mathcal{T}_5, \mathcal{T}_6`.
  - Alternate mobility patterns 1, 2, 3.
- Script: Run `exp4.py`.

## Notes

- Ensure all scripts (`cexp3.py` and `exp4.py`) are correctly configured with the specified parameters before execution.
- Results should be analyzed as described in the manuscript to ensure consistency with the reported findings.
