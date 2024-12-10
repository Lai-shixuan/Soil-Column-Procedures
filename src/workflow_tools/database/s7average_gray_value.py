import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from src.API_functions.Images import file_batch as fb

class GrayValueAnalyzer:
    """Utility class for analyzing average gray values across image stacks"""
    
    @staticmethod
    def average_gray_value(path_in: str):
        """Calculate average gray values from stack of images"""
        # Get image names
        images_paths = fb.get_image_names(path_in, None, 'tif')
        
        # Pre-allocate results list
        results = []
        total_images = len(images_paths)
        
        # Process images one by one
        for idx, img_path in enumerate(images_paths, 1):
            if idx % 500 == 0:  # Progress feedback every 500 images
                print(f"Processing image {idx}/{total_images}")
                
            # Read single image
            img = fb.read_images([img_path], 'gray', read_all=False)[0]

            # Add a mask to exclude background 0 values
            mask = img > 0
            img = img[mask]

            # Calculate and store average
            results.append(np.mean(img))
            
        return results

    @classmethod
    def process_multiple_columns(cls, base_path: str, output_dir: str, columns: list, column_pairs=None):
        """Process specific soil columns and save combined results
        Args:
            columns: List of column numbers to process [5, 7, 9]
            column_pairs: List of tuples specifying which columns should be paired in same colors
        """
        os.makedirs(output_dir, exist_ok=True)
        all_data = {}
        max_length = 0
        global_min = float('inf')
        global_max = float('-inf')
        
        # Collect all data and find global min/max
        for col_num in columns:
            col_id = f"Soil.column.00{col_num:02d}"
            
            print(f"\nProcessing {col_id}...")
            try:
                avg_values = cls.average_gray_value(os.path.join(base_path, col_id, '3.Harmonized/image'))
                all_data[col_id] = avg_values
                max_length = max(max_length, len(avg_values))
                
                # Update global min/max
                values = np.array(avg_values)
                global_min = min(global_min, np.min(values))
                global_max = max(global_max, np.max(values))
                print(f"Completed {col_id} - {len(avg_values)} slices")
            except Exception as e:
                print(f"Error processing {col_id}: {str(e)}")
                continue

        # Normalize lengths with NaN padding
        for col_id in all_data:
            current_length = len(all_data[col_id])
            if current_length < max_length:
                padding = [np.nan] * (max_length - current_length)
                all_data[col_id] = all_data[col_id] + padding

        # Define plot parameters
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        # Create plots with consistent axes
        plt.figure(figsize=(15, 8))  # Combined plot

        # Determine color assignment strategy
        if len(columns) == 1:
            # Single column - use black
            color_map = {columns[0]: ('black', '-')}
        elif column_pairs:
            # Paired columns
            color_map = {}
            for idx, (col1, col2) in enumerate(column_pairs):
                color_idx = idx % len(colors)
                color_map[col1] = (colors[color_idx], '-')
                color_map[col2] = (colors[color_idx], '--')
        else:
            # Multiple columns - different colors
            color_map = {
                col: (colors[idx % len(colors)], '-')
                for idx, col in enumerate(columns)
            }

        # Plot individual and combined graphs
        for col_num in columns:
            col_id = f"Soil.column.00{col_num:02d}"
            if col_id not in all_data:
                continue
                
            values = all_data[col_id]
            color, linestyle = color_map[col_num]
            
            # Individual plot with consistent axes
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(values)), values, color=color)
            plt.title(f'Average Gray Value - {col_id}')
            plt.xlabel('Slice Number')
            plt.ylabel('Average Gray Value')
            plt.grid(True)
            plt.ylim(global_min, global_max)
            plt.savefig(os.path.join(output_dir, f'{col_id}_gray_values.png'))
            plt.close()

            # Add to combined plot
            plt.figure(1)
            plt.plot(range(len(values)), values, 
                    label=col_id,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2)

        # Save summary CSV
        df = pd.DataFrame(all_data)
        df.index.name = 'slice_number'
        csv_path = os.path.join(output_dir, 'combined_gray_values.csv')
        df.to_csv(csv_path)

        # Finalize combined plot
        plt.figure(1)
        plt.title('Average Gray Values - All Columns (Paired Colors)')
        plt.xlabel('Slice Number')
        plt.ylabel('Average Gray Value')
        plt.grid(True)
        plt.ylim(global_min, global_max)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_gray_values.png'), 
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

        return csv_path

def main():
    # Configuration
    config = {
        'base_path': "f:/3.Experimental_Data/Soils/Dongying_normal/",
        'output_dir': "f:/3.Experimental_Data/Soils/Dongying_normal/Analysis/",
        'columns': [i for i in range(22, 28)],  # Specify exact columns to process
        'column_pairs': None    # [(i, i+6) for i in range(10, 16)]
    }
    
    print("Configuration:")
    print(f"Base path: {config['base_path']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Processing columns: {config['columns']}")
    print(f"Column pairs: {config['column_pairs']}\n")
    
    analyzer = GrayValueAnalyzer()
    csv_path = analyzer.process_multiple_columns(
        base_path=config['base_path'],
        output_dir=config['output_dir'],
        columns=config['columns'],
        column_pairs=config['column_pairs']
    )
    print(f"\nResults saved in: {config['output_dir']}")
    print(f"Combined CSV: {csv_path}")

if __name__ == '__main__':
    main()