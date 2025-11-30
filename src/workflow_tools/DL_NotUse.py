## Not use functions, to be stored here.

def deal_with_nan(epoch, model_output):
    """Deal with NaN values in model output."""
    if torch.isnan(model_output).any():
    
        model_output_no_nan = torch.nan_to_num(model_output, nan=0.0)
        mean_value = model_output_no_nan.mean()
        model_output = torch.where(torch.isnan(model_output), mean_value, model_output)

        nan_count = torch.sum(torch.isnan(model_output))
        print(f"In {epoch}, Warning: {nan_count} NaN values in model_output.")

    return model_output

def fetch_unlabeled_batch(unlabeled_iter, unlabeled_loader):
    """Fetches a batch from the unlabeled data loader. If the iterator is exhausted, it resets the iterator and changes the transform to geometric."""
    try:
        batch, mask = next(unlabeled_iter)
    except StopIteration:
        unlabeled_iter = iter(unlabeled_loader)
        batch, mask = next(unlabeled_iter)
    return batch, mask, unlabeled_iter


def update_teacher_model(teacher_model, student_model, alpha):
    """Update teacher model by exponential moving average of student weights."""
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data

def compute_consistency_loss(student_model, teacher_model, device, transform_train,
                            images, masks,
                            epoch, rampup_weight, criterion, threshold=0.8):

    with torch.no_grad():
        output = teacher_model(images)

    output = deal_with_nan(epoch, output)
    teacher_pred = torch.sigmoid(output).squeeze(1)

    threshold = threshold * rampup_weight
    confs = torch.where((teacher_pred > threshold) | (teacher_pred < 1 - threshold), 1, 0).float()

    teacher_pred = (teacher_pred > 0.5).float()

    batch_imgs = []
    batch_labels = []
    batch_masks = []
    batch_conf = []
    for img, label, mask, conf in zip(images, teacher_pred, masks, confs):
        img_np = img.squeeze(0).cpu().numpy()
        label_np = label.cpu().numpy()
        mask_np = mask.cpu().numpy()
        conf_np = conf.cpu().numpy()

        augmenter = s4augmented_labels.ImageAugmenter(img_np, label_np, additional_img=conf_np, mask=mask_np)
        augmented_img, augmented_label, augmented_conf = augmenter.augment()

        augmented = transform_train(image=augmented_img, masks=[augmented_label, mask_np, augmented_conf])
        batch_imgs.append(augmented['image'])
        batch_labels.append(augmented['masks'][0])
        batch_masks.append(augmented['masks'][1])
        batch_conf.append(augmented['masks'][2])

    trans_imgs = torch.stack(batch_imgs).to(device, non_blocking=True)
    trans_lbls = torch.stack(batch_labels).to(device, non_blocking=True)
    trans_masks = torch.stack(batch_masks).to(device, non_blocking=True)
    trans_conf = torch.stack(batch_conf).to(device, non_blocking=True)

    trans_masks = trans_conf * trans_masks

    student_pred = student_model(trans_imgs).squeeze(1)
    loss = criterion(student_pred, trans_lbls, trans_masks)
    
    return loss * rampup_weight


def calculate_update(
    soft_dice_list, epoch, device, model, train_dataset, val_dataset, my_parameters):
    """Calculate update based on soft dice scores"""
    soft_dice_array = np.stack(soft_dice_list)
    train_update_path, val_update_path = dl_config.get_image_output_paths()
    update_status = cvat_nosiy.UpdateStrategy.if_update(soft_dice_array, epoch, threshold=0.9)

    if update_status:
        if my_parameters['mode'] == 'supervised':
            train_eval_loader = DataLoader(
                train_dataset,
                batch_size=my_parameters['label_batch_size'],
                shuffle=False
            )
            train_dataset.set_use_transform(False)
            for batch_idx, (imgs, lbls, msks) in enumerate(tqdm(train_eval_loader)):
                imgs = imgs.to(device)
                imgs = imgs.unsqueeze(1)
                with torch.no_grad():
                    preds = torch.sigmoid(model(imgs))
                    preds = preds.squeeze(1)
                for i in range(len(preds)):
                    dataset_idx = batch_idx * train_eval_loader.batch_size + i
                    train_dataset.update_label_by_index(dataset_idx, preds[i], threshold=0.8)

            val_eval_loader = DataLoader(
                val_dataset,
                batch_size=my_parameters['label_batch_size'],
                shuffle=False
            )
            val_dataset.set_use_transform(False)
            for batch_idx, (imgs, lbls, msks) in enumerate(tqdm(val_eval_loader)):
                imgs = imgs.to(device)
                imgs = imgs.unsqueeze(1)
                with torch.no_grad():
                    preds = torch.sigmoid(model(imgs))
                    preds = preds.squeeze(1)
                for i in range(len(preds)):
                    dataset_idx = batch_idx * val_eval_loader.batch_size + i
                    val_dataset.update_label_by_index(dataset_idx, preds[i], threshold=0.8)

            # Print label stats for selected indices
            sample_indices = range(0, 101, 10)
            for idx in sample_indices:
                if idx < len(train_dataset.labels):
                    label_array = train_dataset.labels[idx]
                    cv2.imwrite(train_update_path / f'{idx}-{epoch}.tif', label_array)
                if idx < len(val_dataset.labels):
                    label_array = val_dataset.labels[idx]
                    cv2.imwrite(val_update_path / f'{idx}-{epoch}.tif', label_array)

            train_dataset.set_use_transform(True)
            val_dataset.set_use_transform(True)
        print(f"Update at epoch {epoch}")
    return update_status
