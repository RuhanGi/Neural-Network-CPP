import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def main():
    if len(sys.argv) < 4:
        print("Usage: python plot_results.py history.csv graph.csv complexity.csv")
        return

    # Load data
    try:
        history_df = pd.read_csv(sys.argv[1])
        graph_df = pd.read_csv(sys.argv[2])
        complexity_df = pd.read_csv(sys.argv[3])
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    fig = plt.figure(figsize=(14, 10))
    base_name = os.path.basename(sys.argv[1]).replace('_history.csv', '')
    plt.suptitle(f"Neural Network Analysis: {base_name}", fontsize=16)

    # Close on ESC
    def on_key(event):
        if event.key == 'escape':
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 1. Plot History: Loss
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', color='blue', linestyle='--')
    ax1.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', color='red')
    ax1.set_title("Loss vs Iterations")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Plot History: Metric (Acc or R^2)
    ax2 = plt.subplot(2, 2, 2)
    # Check graph_df columns to determine metric name
    is_regression = 'truth' in graph_df.columns
    metric_label = "R^2" if is_regression else "Accuracy"
    
    ax2.plot(history_df['epoch'], history_df['train_metric'], label=f'Train {metric_label}', color='green', linestyle='--')
    ax2.plot(history_df['epoch'], history_df['val_metric'], label=f'Val {metric_label}', color='darkgreen')
    ax2.set_title(f"{metric_label} vs Iterations")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel(metric_label)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Middle Graph: Confusion Matrix OR Prediction vs Truth
    ax3 = plt.subplot(2, 2, 3)
    if is_regression:
        # Regression: Scatter Plot
        ax3.scatter(graph_df['truth'], graph_df['pred'], alpha=0.6, color='purple', edgecolors='w')
        # Draw ideal y=x line
        low = min(graph_df['truth'].min(), graph_df['pred'].min())
        high = max(graph_df['truth'].max(), graph_df['pred'].max())
        ax3.plot([low, high], [low, high], 'k--', alpha=0.7, label='Ideal')
        ax3.set_title("Regression: Prediction vs Truth")
        ax3.set_xlabel("Actual Values")
        ax3.set_ylabel("Predicted Values")
        ax3.legend()
    else:
        # Classification: Heatmap (Confusion Matrix)
        # Note: If classification, your C++ exported it without headers. 
        # Re-read without headers if 'truth' isn't found.
        cf_data = pd.read_csv(sys.argv[2], header=None)
        sns.heatmap(cf_data, annot=True, fmt='d', cmap='YlGnBu', ax=ax3, cbar=False)
        ax3.set_title("Classification: Confusion Matrix")
        ax3.set_xlabel("Predicted Class")
        ax3.set_ylabel("Actual Class")

    # 4. Complexity Analysis (Dual Axis)
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(complexity_df['nodes'], complexity_df['train_metric'], color='green', linestyle='--', label='Train Score')
    ax4.plot(complexity_df['nodes'], complexity_df['val_metric'], color='darkgreen', label='Val Score')
    ax4.set_xlabel("Number of Hidden Nodes")
    ax4.set_ylabel(metric_label, color='green')
    ax4.tick_params(axis='y', labelcolor='green')

    ax4_twin = ax4.twinx()
    ax4_twin.plot(complexity_df['nodes'], complexity_df['train_loss'], color='red', linestyle='--', label='Train Loss')
    ax4_twin.plot(complexity_df['nodes'], complexity_df['val_loss'], color='darkred', label='Val Loss')
    ax4_twin.set_ylabel("Loss", color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    ax4.set_title("Complexity: Performance vs Nodes")
    ax4.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Done! Press ESC to exit.")
    plt.show()

if __name__ == "__main__":
    main()