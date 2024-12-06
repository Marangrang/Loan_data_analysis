import numpy as np
import pandas as pd
import os

# reading data files
# using encoding = "ISO-8859-1" to avoid pandas encoding error
borrowers = pd.DataFrame(pd.read_csv( "data/Loan_tape_data/borrowers.csv", encoding = "LATIN-1", low_memory=False))

# reading data files
# using encoding = "ISO-8859-1" to avoid pandas encoding error
payments = pd.DataFrame(pd.read_csv( "data/Loan_tape_data/payments.csv", encoding = "LATIN-1", low_memory=False))

# reading data files
# using encoding = "ISO-8859-1" to avoid pandas encoding error
loans = pd.DataFrame(pd.read_csv( "data/Loan_tape_data/loans.csv", encoding = "LATIN-1", low_memory=False))

# converting all borrower_id to lowercase
borrowers['borrower_id'] = borrowers['borrower_id'].str.lower()

# converting column to lowercase
loans['borrower_id'] = loans['borrower_id'].str.lower()

# borrowers present in borrowers df but not in loans df
borrowers.loc[~borrowers['borrower_id'].isin(loans['borrower_id']), :]

# Flag missing borrowers in Loans
loans['borrower_missing'] = ~loans['borrower_id'].isin(borrowers['borrower_id'])

# Exclude missing borrowers
filtered_loans = loans[loans['borrower_missing'] == False]

# Convert loan_id to lowercase for uniformity
payments['loan_id'] = payments['loan_id'].str.lower()
filtered_loans.loc[:, 'loan_id'] = filtered_loans['loan_id'].str.lower()

# Flag missing loans in payments
payments['loan_missing'] = ~payments['loan_id'].isin(filtered_loans['loan_id'])

# Exclude missing loans
filtered_payments = payments[payments['loan_missing'] == False]

# Ensure loan_id columns are in lowercase for uniformity
filtered_loans['loan_id'] = filtered_loans['loan_id'].str.lower()
payments['loan_id'] = payments['loan_id'].str.lower()

# Filter loans that exist in the payments dataset
loans_with_payments = filtered_loans[filtered_loans['loan_id'].isin(payments['loan_id'])]
loans_with_payments.head()

# Write-Off Rate
# Assuming `write_off_date` and `write_off_amount` columns are present
# Ensure the 'write_off_date' is in datetime format
filtered_loans.loc[:, 'write_off_date'] = pd.to_datetime(filtered_loans['write_off_date'], errors='coerce')
filtered_loans.loc[:, 'write_off_amount'] = filtered_loans['write_off_amount'].fillna(0)

# Filter out loans that have a write-off date (i.e., loans that are actually written off)
written_off_loans = filtered_loans[filtered_loans['write_off_date'].notna()]

# Extract the month and year from the 'write_off_date'
written_off_loans['write_off_month'] = written_off_loans['write_off_date'].dt.to_period('M')

# Group by 'write_off_month' and calculate the total write-off amount for each month
monthly_write_off = written_off_loans.groupby('write_off_month')['write_off_amount'].sum().reset_index(name='write_off_amount')

# Calculate the total outstanding amount for all loans (constant over time)
total_outstanding = filtered_loans['total_outstanding'].sum()

# Calculate the write-off rate for each month
monthly_write_off['write_off_rate'] = (monthly_write_off['write_off_amount'] / total_outstanding) * 100

total_written_off_amount = filtered_loans['write_off_amount'].fillna(0).sum()
total_loan_amount = filtered_loans['total_outstanding'].sum()

# Calculate Write-Off Rate
filtered_loans['write_off_rate'] = filtered_loans['write_off_amount'] / filtered_loans['total_outstanding']

write_off_rate = (total_written_off_amount / total_loan_amount) * 100

print(f"Write-Off Rate: {write_off_rate:.2f}%")

# Calculate Write-Off Rate by Product
write_off_by_product = filtered_loans.groupby('product_name').agg(
    total_outstanding=('total_outstanding', 'sum'),
    total_write_off=('write_off_amount', 'sum')
).reset_index()

write_off_by_product['write_off_rate'] = (
    write_off_by_product['total_write_off'] / write_off_by_product['total_outstanding']
) * 100


# Collections Rate
total_collections = filtered_loans['principal_amount'].sum()
total_outstanding = filtered_loans['total_outstanding'].sum()

collections_rate = (total_collections / total_outstanding) * 100
print(f"Collections Rate: {collections_rate:.2f}%")

# Gross Yield
total_interest_collected = (filtered_loans['interest_rate'] * filtered_loans['principal_amount']).sum()

# Calculate the total interest and fees (sum of interest rate applied to principal + any penalties or fees)
total_interest_and_fees = (filtered_loans['principal_amount'] * filtered_loans['interest_rate']).sum() + filtered_loans['penalties'].sum() + filtered_loans['fees'].sum()

# Calculate the total principal amount (sum of all principal amounts)
total_principal = filtered_loans['principal_amount'].sum()

# Calculate Gross Yield
gross_yield = (total_interest_and_fees / total_principal) * 100
print(f"Gross Yield: {gross_yield:.2f}%")

# Average Days in Arrears
# Assuming 'filtered_loans' is your DataFrame and 'as_of_datetime' is in datetime format
filtered_loans.loc[:, 'as_of_datetime'] = pd.to_datetime(filtered_loans['as_of_datetime'])

# Calculate days in arrears
def calculate_days_in_arrears(row):
    if pd.notna(row['default_date']):
        arrears_date = pd.to_datetime(row['default_date'])
    elif pd.notna(row['write_off_date']):
        arrears_date = pd.to_datetime(row['write_off_date'])
    else:
        # Use 'maturity_date' if no default or write-off date exists
        arrears_date = pd.to_datetime(row['maturity_date'])

    # Calculate the difference between 'as_of_datetime' and arrears date
    days_in_arrears = (row['as_of_datetime'] - arrears_date).days

    return max(days_in_arrears, 0)  # Avoid negative days in arrears

# Apply the function to calculate days in arrears for each loan
filtered_loans['days_in_arrears'] = filtered_loans.apply(calculate_days_in_arrears, axis=1)

# Calculate the average days in arrears
average_days_in_arrears = filtered_loans['days_in_arrears'].mean()

# Output the result
print(f"Average Days in Arrears: {average_days_in_arrears:.2f}")

# Portfolio At Risk (PAR)
# Convert 'as_of_datetime' and 'maturity_date' to datetime if they are not already
filtered_loans['as_of_datetime'] = pd.to_datetime(filtered_loans['as_of_datetime'], errors='coerce')
filtered_loans['maturity_date'] = pd.to_datetime(filtered_loans['maturity_date'], errors='coerce')

# Define threshold for PAR calculation (e.g., loans overdue > 854 days)
par_threshold = 854

# Calculate loans at risk
loans_at_risk = filtered_loans[
    (filtered_loans['as_of_datetime'] - filtered_loans['maturity_date']).dt.days > par_threshold
]

# Calculate Portfolio At Risk
par = (loans_at_risk['principal_amount'].sum() / filtered_loans['principal_amount'].sum()) * 100

print(f"Portfolio At Risk (PAR): {par:.2f}%")

# Recovery Rate
# Filter recoveries from payment data
recovery_payments = filtered_payments[filtered_payments['type'] == 'RECOVERIES']

recovery_payments['payment_date'] = pd.to_datetime(recovery_payments['payment_date'], errors='coerce')

# Add payment month
recovery_payments['payment_month'] = recovery_payments['payment_date'].dt.to_period('M')

# Group by payment month to calculate total recoveries for each month
monthly_recoveries = recovery_payments.groupby('payment_month').agg(
    total_recoveries=('amount', 'sum')
).reset_index()

# Calculate total recoveries
total_recoveries = recovery_payments['amount'].sum()

# Ensure write_off_amount is not NaN
filtered_loans['write_off_amount'] = filtered_loans['write_off_amount'].fillna(0)

# Add month column to loans to group by month
filtered_loans['write_off_month'] = filtered_loans['write_off_date'].dt.to_period('M')

# Group by month to calculate total write-offs for each month
monthly_write_offs = filtered_loans.groupby('write_off_month').agg(
    total_write_offs=('write_off_amount', 'sum')
).reset_index()

# Merge the DataFrames
monthly_recoveries.rename(columns={'payment_month': 'payment_month'}, inplace=True)
monthly_write_offs.rename(columns={'write_off_month': 'payment_month'}, inplace=True)

# Merge the recovery and write-off data on the payment month (same as write-off month)
monthly_recovery_data = pd.merge(monthly_recoveries, monthly_write_offs, on='payment_month', how='inner')

# Calculate the recovery rate for each month
monthly_recovery_data['recovery_rate'] = (monthly_recovery_data['total_recoveries'] / monthly_recovery_data['total_write_offs']) * 100

# Calculate total write-offs
total_write_offs = filtered_loans['write_off_amount'].sum()

# Calculate Recovery Rate
recovery_rate = (total_recoveries / total_write_offs) * 100 if total_write_offs > 0 else 0

print(f"Recovery Rate: {recovery_rate:.2f}%")

# Penalty Rate
# Calculate Penalty Rate
penalty_rate = (filtered_loans['penalties'].sum() / filtered_loans['principal_amount'].sum()) * 100

print(f"Penalty Rate: {penalty_rate:.2f}%")

# Active Loans
# Filter active loans
active_loans = filtered_loans[filtered_loans['closing_date'].isna()]

# Count and total amount of active loans
active_loan_count = active_loans['loan_id'].nunique()
active_loan_total = active_loans['principal_amount'].sum()
print(f"Active Loans: {active_loan_count} loans, Total Amount: {active_loan_total:.2f}")

# Loan Repayment Rate
# Fully repaid loans
fully_repaid_loans = filtered_loans[(filtered_loans['total_outstanding'] == 0) & (filtered_loans['closing_date'] <= filtered_loans['maturity_date'])]

# Calculate Repayment Rate
repayment_rate = (fully_repaid_loans['principal_amount'].sum() / filtered_loans['principal_amount'].sum()) * 100
print(f"Loan Repayment Rate: {repayment_rate:.2f}%")

# Default Rate
# 1. Filter for Defaulted Loans
filtered_loans['default_date'] = pd.to_datetime(filtered_loans['default_date'], errors='coerce')  # Convert to datetime
default_loans = filtered_loans[filtered_loans['default_date'].notna()]

# 2. Extract the Month and Year from the default date
default_loans['default_month'] = default_loans['default_date'].dt.to_period('M')

# 3. Group by Month to calculate monthly defaults
monthly_defaults = default_loans.groupby('default_month').size().reset_index(name='defaults')

# 4. Calculate the Total Loans (constant, as the dataset is filtered)
total_loans = len(filtered_loans)

# 5. Calculate Default Rate per Month
monthly_defaults['default_rate'] = (monthly_defaults['defaults'] / total_loans) * 100

default_rate = (len(default_loans) / total_loans) * 100

print(f"Default Rate: {default_rate:.2f}%")

default_rate_by_product = (
    filtered_loans.groupby('product_name')['default_date']
    .apply(lambda x: x.notna().sum() / len(x) * 100)
)
print(default_rate_by_product)

# Delinquency Rate
# Assuming 'filtered_loans' is your dataset
# Step 1: Ensure you have necessary columns and convert dates to datetime
filtered_loans['maturity_date'] = pd.to_datetime(filtered_loans['maturity_date'])
filtered_loans['default_date'] = pd.to_datetime(filtered_loans['default_date'])
filtered_loans['write_off_date'] = pd.to_datetime(filtered_loans['write_off_date'])
filtered_loans['as_of_datetime'] = pd.to_datetime(filtered_loans['as_of_datetime'])

# Step 2: Define delinquent loans
delinquent_loans = filtered_loans[
    (filtered_loans['maturity_date'] < filtered_loans['as_of_datetime']) &  # Loan is overdue
    (filtered_loans['default_date'].isna()) &  # No default date
    (filtered_loans['write_off_date'].isna())  # No write-off date
]

# We will create a new column to mark each loan as delinquent or not based on its maturity date
filtered_loans['is_delinquent'] = (filtered_loans['maturity_date'] < filtered_loans['as_of_datetime']) & \
                                  (filtered_loans['default_date'].isna()) & \
                                  (filtered_loans['write_off_date'].isna())

# Step 3: Calculate the Delinquency Rate
total_loans = len(filtered_loans)
delinquent_loans_count = len(delinquent_loans)

payments['payment_date'] = pd.to_datetime(payments['payment_date'])

# Group payments by month (you can change the period to 'quarter' or 'year' if needed)
payments['payment_month'] = payments['payment_date'].dt.to_period('M')

# Create a dataframe to track delinquent loans over time
delinquent_loans_over_time = payments.groupby('payment_month').apply(
    lambda x: (filtered_loans['is_delinquent'] & filtered_loans['loan_id'].isin(x['loan_id'])).sum()
).reset_index(name='delinquent_loans')

# Step 4: Calculate total loans at each period
total_loans_over_time = payments.groupby('payment_month').apply(
    lambda x: len(filtered_loans[filtered_loans['loan_id'].isin(x['loan_id'])])
).reset_index(name='total_loans')

# Merge delinquent loans and total loans
delinquency_data = pd.merge(delinquent_loans_over_time, total_loans_over_time, on='payment_month')

# Step 5: Calculate delinquency rate per period
delinquency_data['delinquency_rate'] = (delinquency_data['delinquent_loans'] / delinquency_data['total_loans']) * 100

delinquency_rate = (delinquent_loans_count / total_loans) * 100

# Output the delinquency rate
print(f"Delinquency Rate: {delinquency_rate:.2f}%")