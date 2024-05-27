import API_functions.file_batch as fb
import API_functions.Soils.column_batch_process as column_bp


def start(mode: str, column_path: str, column_id: int, part_nums: int, parts_range: list, fill_in_folder: list, start_index: int, middile_name: str):

    def check_continuous(lst):
        sorted_lst = sorted(lst)
        is_continuous = all(sorted_lst[i] + 1 == sorted_lst[i + 1] for i in range(len(sorted_lst) - 1))
        
        if is_continuous:
            return [min(sorted_lst)]
        else:
            return lst

    my_column = column_bp.SoilColumn(column_path, column_id=column_id, part_num=part_nums)
    read_from_folder = [x - 1 for x in fill_in_folder]
    read_from_folder = check_continuous(read_from_folder)
    check_roi = True if 3 in fill_in_folder else False


    if mode == "make_dirs":
        pass

    if mode == "check":
        check_values(my_column, parts_range, read_from_folder, check_roi)

    elif mode == "process":
        check_values(my_column, parts_range, read_from_folder, check_roi)
        for i in parts_range:
            # Read first
            my_roi = my_column.get_part(part_index=i).get_roi()
            paths = []
            for j in range(6):
                paths[j] = my_column.get_part(part_index=i).get_subfolder_path(index=j)

            # Process later
            image_write_name = fb.ImageName(prefix=f"{my_column.column_id}-{i:2d}_{middile_name}", suffix='')

            fb.roi_select(path_in=paths[1], path_out=paths[2], name_read=None, roi=my_roi)
            fb.rename(path_in=paths[2], path_out=paths[3], new_name=image_write_name, start_index=start_index, reverse=True)
            fb.image_process(path_in=paths[3], path_out=paths[4])

            start_index += len(fb.get_image_list(paths[3]))


def check_values(my_column, part_ids, step_indices, check_roi):
    def get_selected_steps(step_indices):
        steps = ['0.Origin', '1.Reconstruct', '2.ROI', '3.Rename', '4.Threshold', '5.Analysis']
        selected_steps = [steps[i] for i in step_indices if i < len(steps)]
        
        if check_roi:
            selected_steps.extend(['ROI File Exists', 'ROI File Filled'])
        return selected_steps

    def format_part_ids(part_ids):
        return [f"{int(part_id):02d}" for part_id in part_ids]

    def fill_data_rows(data_structure, part_ids, selected_steps):
        # Prepare the rows for each header
        rows = {header: [] for header in selected_steps}
        rows["Part ID"] = [key for key in data_structure if key.split('-')[-1] in part_ids]
        for key in rows["Part ID"]:
            value = data_structure[key]
            for header in selected_steps[1:]:
                if 'ROI File' in header:
                    data = value.get('ROI File', {}).get(header.split(' ')[-1].lower(), '[N/A]')
                else:
                    data = value.get(header, '[N/A]')
                
                if isinstance(data, bool):
                    data = "True" if data else "[False]"
                elif data == 0 or data == 'No':
                    data = f"[{data}]"
                rows[header].append(data)
        return rows

    def print_data_table(headers, rows):
        # Print headers
        print("")
        print("The missing data will be marked as '[0]', '[False]', or '[N/A]'.")
        print(''.join(f"{header:<15}" for header in headers))
        # Print data rows
        for i in range(len(rows["Part ID"])):
            print(''.join(f"{rows[header][i]:<15}" for header in headers))

    data_structure = my_column.get_data_structure()
    headers = ["Part ID"] + get_selected_steps(step_indices)
    part_ids = format_part_ids(part_ids)
    rows = fill_data_rows(data_structure, part_ids, headers)
    print_data_table(headers, rows)

    # if the data structure is not complete, raise an error
    for header in headers[1:]:
        if '[0]' in rows[header] or ['N/A'] in rows[header] or ['False'] in rows[header]:
            raise ValueError(f"Data structure is not complete.")


if __name__ == "__main__":
    params = {
        # Basic parameters
        'mode': 'make_dirs',  # "check", "make_dirs", or "process"
        'column_path': "e:/3.Experimental_Data/Soil.column.0002/",
        'column_id': 2,
        'part_nums': 5,

        # The parameters below are used in the "check" or "process" mode
        'parts_range': [1],
        'fill_in_folder': [2, 3, 4],
        'start_index': 0,
        'middile_name': "Ou_DongYing"
    }
    start(**params)