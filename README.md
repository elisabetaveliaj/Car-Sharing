# Case study: Car Sharing Company; Data Analysis and Optimization

Overview
This repository contains the code and documentation for the data analysis and optimization project carried out as part of the Master's program Technology and Operations Management course: Data Analysis and Programming for Operations Management.

Problem Statement
A car-sharing company was facing challenges related to poor customer satisfaction and profitability. The objective of this project was to analyze the provided data from the past year to identify patterns in yearly and daily demand. Additionally, an optimization model was developed to improve the car assignment process, for enhancing fleet utilization and customer satisfaction.

Data Analysis
Process
- Data Cleaning: The provided data file was cleaned using Python and the Elasticsearch client. Invalid errors such as trips with identical start and end times or locations, and unrealistic trips based on average speed were removed.
- Visualization: The cleaned data was visualized to analyze yearly and daily demand patterns in 2023.

Daily Demand Analysis
- Pattern Identification: Daily demand patterns were analyzed to identify peak periods and variability throughout the day.
- Working Day Analysis: Demand on a typical working day was visualized on an hourly basis, along with statistical measures such as mean and standard deviation.
- Weekend Demand: Weekend demand patterns were analyzed for a comprehensive understanding of demand fluctuations.

Optimization Model
Overview
The optimization model aims to improve the car assignment process by strategically assigning cars to customer requests, thereby optimizing fleet utilization and increasing profitability and customer satisfaction.

Mathematical Model
- Objective: Maximize total profit from utilizing car-sharing requests.
- Decision Variable: Binary variable for each potential assignment of a car to a request, indicating whether or not the request is served by that particular car.
- Constraints:
  - Assignment Constraint: Ensure each request is assigned to at most one car.
  - Capacity Constraint: Ensure each car is assigned to at most one request.
  - Walking Distance Constraint: A request can only be assigned to a car if the car is within walking distance from the requester’s origin.
- Profit Calculation: Profit for serving a request is based on the straight-line distance between the origin and destination, multiplied by a revenue factor per started minute.

Implementation
The mathematical model is implemented in Python and Gurobi, utilizing optimization techniques to solve optimal car assignments.

