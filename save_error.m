function save_error(model_name, num_epochs)
  modelpath = fullfile('data',model_name,sprintf('net-epoch-%i.mat', num_epochs))
  model = load(modelpath, 'net', 'info')
  train_error_top1 = model.info.train.error(1,:)
  train_error_top5 = model.info.train.error(2,:)
  val_error_top1 = model.info.val.error(1,:)
  val_error_top5 = model.info.val.error(2,:)
  save(fullfile('error_stores', sprintf('%s-error.mat', model_name)), 'train_error_top1','train_error_top5', 'val_error_top1', 'val_error_top5') ;
end
