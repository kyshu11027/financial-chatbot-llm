from typing import Optional, Literal
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from langchain_core.tools import tool

class PlotConfig(BaseModel):
    plot_type: Literal["line", "bar", "pie", "scatter", "histogram"] = Field(description="Type of plot to create")
    x_axis: str = Field(description="Column for x-axis")
    y_axis: Optional[str] = Field(description="Column for y-axis", default=None)
    title: str = Field(description="Plot title")
    group_by: Optional[str] = Field(description="Column to group by", default=None)

@tool
def create_financial_plot(transactions_json: str, plot_config: PlotConfig) -> str:
    """Create visualizations of financial data.
    
    Args:
        transactions_json: JSON string of transaction data
        plot_config: Configuration for the plot including type, axes, and grouping
    
    Returns:
        Base64 encoded plot image
    """
    try:
        # Convert JSON to DataFrame
        transactions = pd.read_json(transactions_json)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        if plot_config.plot_type == "line":
            if plot_config.group_by:
                for group in transactions[plot_config.group_by].unique():
                    group_data = transactions[transactions[plot_config.group_by] == group]
                    plt.plot(group_data[plot_config.x_axis], group_data[plot_config.y_axis], label=group)
                plt.legend()
            else:
                plt.plot(transactions[plot_config.x_axis], transactions[plot_config.y_axis])
                
        elif plot_config.plot_type == "bar":
            if plot_config.group_by and plot_config.y_axis:
                grouped_data = transactions.groupby(plot_config.group_by)[plot_config.y_axis].sum()
                grouped_data.plot(kind='bar')
            else:
                transactions.plot(kind='bar', x=plot_config.x_axis, y=plot_config.y_axis)
                
        elif plot_config.plot_type == "pie":
            if plot_config.group_by and plot_config.y_axis:
                grouped_data = transactions.groupby(plot_config.group_by)[plot_config.y_axis].sum()
                labels = grouped_data.index.astype(str).tolist()
                plt.pie(grouped_data, labels=labels, autopct='%1.1f%%')
            else:
                labels = transactions[plot_config.x_axis].astype(str).tolist()
                plt.pie(transactions[plot_config.y_axis], labels=labels, autopct='%1.1f%%')
                
        elif plot_config.plot_type == "scatter":
            plt.scatter(transactions[plot_config.x_axis], transactions[plot_config.y_axis])
            
        elif plot_config.plot_type == "histogram":
            plt.hist(transactions[plot_config.x_axis], bins=30)
        
        plt.title(plot_config.title)
        plt.tight_layout()
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{plot_base64}"
        
    except Exception as e:
        return f"Error creating plot: {str(e)}"