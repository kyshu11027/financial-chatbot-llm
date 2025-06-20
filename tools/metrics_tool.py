# import pandas as pd
# import json
# from langchain_core.tools import tool

# @tool
# def calculate_financial_metrics(transactions_json: str, metric_type: str) -> dict:
#     """Calculate financial metrics from transaction data.
    
#     Args:
#         transactions_json: JSON string of transaction data
#         metric_type: Type of metric to calculate (spending_by_category, monthly_trends, budget_analysis)
    
#     Returns:
#         Dictionary with calculated metrics
#     """
#     try:
#         transactions = json.loads(transactions_json)
#         df = pd.DataFrame(transactions)
        
#         if metric_type == "spending_by_category":
#             category_spending = df.groupby('category')['amount'].sum().to_dict()
#             return {
#                 "metric": "spending_by_category",
#                 "data": category_spending,
#                 "total_spending": df['amount'].sum(),
#                 "top_category": max(category_spending, key=lambda k: category_spending[k]) if category_spending else None
#             }
            
#         elif metric_type == "monthly_trends":
#             df['date'] = pd.to_datetime(df['date'])
#             df['month'] = df['date'].dt.to_period('M')
#             monthly_spending = df.groupby('month')['amount'].sum().to_dict()
#             return {
#                 "metric": "monthly_trends",
#                 "data": {str(k): v for k, v in monthly_spending.items()},
#                 "average_monthly": df.groupby('month')['amount'].sum().mean()
#             }
            
#         elif metric_type == "budget_analysis":
#             total_spending = df['amount'].sum()
#             avg_transaction = df['amount'].mean()
#             return {
#                 "metric": "budget_analysis",
#                 "total_spending": total_spending,
#                 "average_transaction": avg_transaction,
#                 "transaction_count": len(df),
#                 "spending_categories": len(df['category'].unique())
#             }
            
#         return {"error": "Unknown metric type"}
        
#     except Exception as e:
#         return {"error": f"Error calculating metrics: {str(e)}"}