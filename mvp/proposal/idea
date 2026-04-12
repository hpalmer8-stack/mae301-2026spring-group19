MAE 301 Project Proposal: Engine Health Analysis
Daniel Pham, Jacob Baxely, Hunter Palmer, Gavin Pessefall, Eric Jocque
Spring 2026


3. Problem Statement

Our product is intended for fleet vehicle managers and vehicle owners who aim to maintain vehicle reliability and performance. Engine failure is an expensive occurrence that causes unexpected downtime, delays, and costly repairs. Although predicting and addressing failure early can reduce the damage and repair costs, early symptoms are often difficult to recognize.

4. Why Now?

Companies are increasingly relying on large fleets of vehicles to transport people, materials, and products. As the cost of new vehicles and repairs continues to rise, vehicle owners will invest in ways to increase vehicle life and prevent unexpected issues. The development of microphones in consumer electronics and advances in machine learning make our product accessible to a wide range of consumers.

5. Proposed AI-Powered Solution

We propose an AI tool that analyzes engine audio to assess its health. It will recognize any symptoms of excessive wear or premature failure and present the most likely diagnosis, along with its severity. It would be trained on a range of vehicle models and conditions, allowing it to match sound patterns with the most likely fault. This tool would increase the efficiency and accuracy of regular vehicle inspections and make failure more predictable. It would also simplify the troubleshooting process, decreasing repair time.
Traditional diagnostic tools rely on simple decibel thresholds or frequency filters, which are less effective in noisy real-world environments. Transformer-based architecture treats engine audio as a temporal sequence, allowing the AI to identify specific rhythmic signatures regardless of background noise or engine RPM.

6. Initial Technical Concept

The data used for this AI model consists of recordings of healthy and unhealthy engine noises, with the goal of training the AI to determine whether an engine is healthy or not when given a recording of our own engines. The goal is to use a classifier model to determine whether an engine is healthy, and then use a GPT-style model to generate a short prediction of what might be wrong with the engine. The work from nanoGPT will be used to output a binary classification between healthy and abnormal, with some fine-tuning to ensure the AI model is extremely accurate and can correctly evaluate a 5-second audio file of a car engine. 

7. Scope for MVP

A realistic MVP for a 6-week project is a model that can recognize whether an engine is unhealthy or healthy without a specific diagnosis. A user can record audio of their vehicle while running, and our system returns a report that either confirms a healthy engine or recommends further inspection.

8. Risks and Open Questions

Data availability:
One risk is that publicly available engine audio datasets may not perfectly match real-world recordings from consumer vehicles. Differences in microphones, recording environments, and engine types could introduce domain mismatch between training data and real user recordings.
Evaluation reliability:
Evaluating model accuracy may be challenging because real-world “abnormal” engine recordings are harder to obtain and verify than healthy ones. This may require careful validation using multiple datasets and controlled recordings.
User adoption:
Another open question is whether users would trust an AI-based diagnostic tool for engine health. Ensuring the system provides clear explanations and confidence scores will be important for building user confidence in the results.

9. Planned Data Sources

We plan to use publicly available engine audio datasets from Kaggle, including datasets focused on engine condition and automotive diagnostic sounds. These datasets will provide labeled recordings of normal and abnormal engine behavior for training the classification model.
Additional testing data will be collected by recording idle engine audio from Toyota trucks owned by team members. This real-world data will be used to validate and demonstrate the MVP.
