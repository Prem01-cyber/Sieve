# Sieve : Customer Support Reinforcement Learning Environment

Primarily there are gonna be three major tasks **Email Classification**, **Response Drafting** and **Support Session**

## Email Classification - Task 1 

The agent here receives one email at a time and must classify into the categories **billing**, **technical**, **general**, **spam**, **account** and **feature_request** and respective urgencies i.e **high**, **medium** and **low**

Rewards shall be assigned for each correct category and urgency classification.

## Response Drafting - Task 2 

After classification now we need to prioritize over drafting the response, so agent reads a customer email and drafts a response in accordance with the classified email. Responses generated will be graded on 3 major factors 

- Covering the relevant information in the response for the issue so that the issue is addressed entirely.
- Not to generate more than required information and over explain with over information.
- Maintain a preferred tone over throughout the response generated.

Rewards shall be assigned based on the coverage and length of the response drafted

## Support Session - Task 3 

The agent manages a queue of mixed emails and must perform below actions 

- Identify and prioritize high priority customers first 
- Handle high urgency emails before low urgency 
- Choosing actions i.e **respond**, **escalate** or **archive** accordingly
- Provide correct category and urgency classification

# TODO: Develop a structure for Observation and Action Space 
