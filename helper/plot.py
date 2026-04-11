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
        # Note: We don't load graph_df here yet; we handle it specifically below
        comp_df = pd.read_csv(sys.argv[3])
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    fig = plt.figure(figsize=(16, 10))
    
    # Extract base name for title
    base_filename = os.path.basename(sys.argv[1])
    base_name = base_filename.split('_')[0] if '_' in base_filename else base_filename.split('.')[0]
    plt.suptitle(f"Neural Network Analysis: {base_name}", fontsize=18)

    # Close on ESC
    def on_key(event):
        if event.key == 'escape': plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 1. History (Loss & Metric vs Iterations)
    #
    ax1 = plt.subplot(2, 2, 1)
    # ... plotting ax1 and ax1_twin ...
    #
    ax1.plot(history_df['epoch'], history_df['train_loss'], color='blue', linestyle='--', label='Train Loss')
    ax1.plot(history_df['epoch'], history_df['val_loss'], color='red', label='Val Loss')
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper left')

    ax1_twin = ax1.twinx()
    
    # Heuristic to detect metric label: Check if final val_metric in complexity is <= 1.0 (Acc/R2)
    # If using history, we can check final val_score. Assuming Accuracy unless R2 is known.
    # In the absence of 'truth' column yet, we assume Acc if <= 1.
    metric_label = "Accuracy" if history_df['val_metric'].max() <= 1.0 else "R^2"

    ax1_twin.plot(history_df['epoch'], history_df['train_metric'], color='green', linestyle='--', label=f'Train {metric_label}')
    ax1_twin.plot(history_df['epoch'], history_df['val_metric'], color='darkgreen', label=f'Val {metric_label}')
    ax1_twin.set_ylabel(metric_label)
    ax1_twin.legend(loc='upper right')
    ax1.set_title("Training History (Loss & Metric)")

    # 2. Confusion Matrix or Regression (SPLIT TRAIN/VAL)
    ax2 = plt.subplot(2, 2, 2)
    
    # Re-reading graph file to handle headers correctly based on content
    raw_graph_data = pd.read_csv(sys.argv[2])

    if 'truth' in raw_graph_data.columns and 'set' in raw_graph_data.columns:
        # REGRESSION: Use Seaborn to split Train/Val
        print("Detected Regression Graph with Train/Val split...")
        
        # Plot Train Points (greyed out and transparent)
        train_pts = raw_graph_data[raw_graph_data['set'] == 'train']
        ax2.scatter(train_pts['truth'], train_pts['pred'], color='grey', alpha=0.2, label='Train Set', s=10)
        
        # Plot Validation Points (colored and solid)
        val_pts = raw_graph_data[raw_graph_data['set'] == 'validation']
        ax2.scatter(val_pts['truth'], val_pts['pred'], color='purple', alpha=0.7, label='Val Set', s=20)

        # Draw ideal y=x line
        low = min(raw_graph_data['truth'].min(), raw_graph_data['pred'].min())
        high = max(raw_graph_data['truth'].max(), raw_graph_data['pred'].max())
        ax2.plot([low, high], [low, high], 'k--', alpha=0.5)
        
        ax2.set_title("Final State: Regression Scatter")
        ax2.set_xlabel("Actual Values")
        ax2.set_ylabel("Predicted Values")
        ax2.legend()
        ax2.grid(True, alpha=0.2)

    elif 'truth' in raw_graph_data.columns:
        # Classic Regression Scatter (No set distinction)
        print("Detected Simple Regression Graph...")
        ax2.scatter(raw_graph_data['truth'], raw_graph_data['pred'], alpha=0.5, color='purple')
        l, h = ax2.get_xlim()[0], ax2.get_xlim()[1]
        ax2.plot([l, h], [l, h], 'k--', alpha=0.6)
        ax2.set_title("Final State: Truth vs Prediction")
        
    else:
        # CLASSIFICATION: Re-reading without headers for standard matrix grid
        print("Detected Classification Graph (Confusion Matrix)...")
        cf_data = pd.read_csv(sys.argv[2], header=None)
        sns.heatmap(cf_data, annot=True, fmt='d', cmap='YlGnBu', ax=ax2, cbar=False)
        ax2.set_title("Final State: Confusion Matrix")
        
    # 3. & 4. Complexity plots remain unchanged
    # ... ...
    ax3 = plt.subplot(2, 2, 3)
    sns.lineplot(data=comp_df, x='nodes', y='val_metric', hue='activation', marker='o', ax=ax3)
    ax3.set_title(f"Complexity: Val {metric_label} vs Nodes")
    ax3.set_ylabel(metric_label)

    ax4 = plt.subplot(2, 2, 4)
    sns.lineplot(data=comp_df, x='nodes', y='val_loss', hue='activation', marker='s', ax=ax4)
    ax4.set_title("Complexity: Val Loss vs Nodes")
    ax4.set_ylabel("Loss")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- SAVE THE IMAGE ---
    save_path = f"plots.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plotting complete!")
    
    plt.show()

if __name__ == "__main__":
    main()