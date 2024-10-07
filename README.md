# Options Trading Environment

Created Options Trading Environment to trade and select options based on current price and price of the asset. Was able to take account from 2,000 to peak of 130,000 dollars. System was set up to send notification to manually enter and exit trade. Attempts at automation as being worked on right now. Results can be seen in `Trading Results` folder


<img src="https://github.com/user-attachments/assets/88dd1190-3346-4dfb-acfd-93f847e2eef7" width="200" />

<img src="https://github.com/user-attachments/assets/543b75f8-b95e-4a6e-97bd-b70254c5015b" width="200" />

Options are difficult to manage as they come and go out of existence as they only live for a limited amount of time. 
To train a model that is capable of handling assets that are moving in and out at such a fast pace, we can organize the 
following table

| Strike Price Delta | 1 day | 2 day | 3 day | ... | N day |
|-------------------|-------|-------|-------|-----|-------|
| -3                | 1.00  | 2.00  | 3.00  |...| 10.00 |
| -2                | 1.10  | 2.80  | 3.90  |...| 10.70 |
| -1                | 1.20  | 2.70  | 3.00  |...| 10.30 |
| 0                 | 1.30  | 2.60  | 3.10  |...| 10.90 |
| 1                 | 1.40  | 2.50  | 3.20  |...| 10.40 |

Where we keep track of the relative attributes. The values will be shift around and automatically sold once no longer tracked.
The relative window can be modified based on handle a larger timeframe or strike price delta.

