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
        # Get image names and read all images
        images_paths = fb.get_image_names(path_in, None, 'png')
        img_stack = fb.read_images(images_paths, 'gray', read_all=True)
        img_stack = np.array(img_stack)
        
        # Calculate average gray value for each slice
        return [np.mean(img_stack[i]) for i in range(img_stack.shape[0])]

    @classmethod
    def process_multiple_columns(cls, base_path: str, output_dir: str, start_col: int, end_col: int):
        """Process multiple soil columns and save combined results"""
        os.makedirs(output_dir, exist_ok=True)
        all_data = {}
        max_length = 0
        global_min = float('inf')
        global_max = float('-inf')
        
        # Collect all data and find global min/max
        for col_num in range(start_col, end_col + 1):
            col_id = f"Soil.column.00{col_num:02d}"
            input_path = os.path.join(base_path, col_id, "2.ROI")
            
            print(f"\nProcessing {col_id}...")
            try:
                avg_values = cls.average_gray_value(input_path)
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
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
        line_styles = ['-', '--']
        mid_point = start_col + (end_col - start_col + 1) // 2

        # Create plots with consistent axes
        plt.figure(figsize=(15, 8))  # Combined plot

        # Plot individual and combined graphs
        for col_num in range(start_col, end_col + 1):
            col_id = f"Soil.column.00{col_num:02d}"
            if col_id not in all_data:
                continue
                
            values = all_data[col_id]
            
            # Individual plot with consistent axes
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(values)), values)
            plt.title(f'Average Gray Value - {col_id}')
            plt.xlabel('Slice Number')
            plt.ylabel('Average Gray Value')
            plt.grid(True)
            plt.ylim(global_min, global_max)
            plt.savefig(os.path.join(output_dir, f'{col_id}_gray_values.png'))
            plt.close()

            # Add to combined plot
            plt.figure(1)
            color_idx = (col_num - start_col) % ((end_col - start_col + 1) // 2)
            style_idx = 0 if col_num < mid_point else 1
            plt.plot(range(len(values)), values, 
                    label=col_id,
                    color=colors[color_idx],
                    linestyle=line_styles[style_idx],
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
        'base_path': "f:/3.Experimental_Data/Soils/Dongying_Tiantan-Hospital/",
        'output_dir': "f:/3.Experimental_Data/Soils/Dongying_Tiantan-Hospital/Analysis/",  # New output directory
        'columns': {
            'start': 10,
            'end': 21
        }
    }
    
    print("Configuration:")
    print(f"Base path: {config['base_path']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Processing columns: {config['columns']['start']} to {config['columns']['end']}\n")
    
    analyzer = GrayValueAnalyzer()
    csv_path = analyzer.process_multiple_columns(
        base_path=config['base_path'],
        output_dir=config['output_dir'],
        start_col=config['columns']['start'],
        end_col=config['columns']['end']
    )
    print(f"\nResults saved in: {config['output_dir']}")
    print(f"Combined CSV: {csv_path}")

if __name__ == '__main__':
    main()