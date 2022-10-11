import os
imort torch

def run_script():
  # Constants
  DATAPATH = os.path.join(os.getcwd(), 'data')
  device = "cuda" if torch.cuda.is_available() else "cpu"
  #print(f"Using {device} device")

  # X: sample x dims
  X_train = np.load(os.path.join(DATAPATH,'X_train.npy'))
  Ydf = pd.read_csv(os.path.join(DATAPATH,'y_train.csv'))
  # y: samples
  Y_train = Ydf.Predicted.to_numpy()
  nbr_classes = len(np.unique(Y_train))
  nbr_dims = len(X_train[0])

  # Init
  with wandb.init() as run:
    config = wandb.config
    print("Config: ", config)
    n_boots = config.general['n_boots']
    n_ensambles = config.general['n_ensambles']
    n_epochs = config.general['n_epochs']
    batch_size =  config.general['batch_size']
    # Pre-set
    spliter = getattr(spliters, config.spliter['name'])(X_train, Y_train,
                                                     **config.spliter.get('kwargs', dict()))
    # Values to record
    iteration, train_loss, val_loss, val_f1 = [], [], [], []
    val_targets, val_predictions, val_certainty = [], [], []
    # Run configuration

    
    for boot_i in range(n_boots):
      # split data
      x_train, x_val, y_train, y_val = spliter.split()
      # Count number of instance in each class
      train_count = [0 for _ in range(nbr_classes)]
      for label in y_train:
        train_count[label] += 1 
        
      val_count = [0 for _ in range(nbr_classes)]
      for label in y_val:
        val_count[label] += 1


      # Initialize Data loader
      
      train_dataset = getattr(datasets, config.dataset['name'])(x_train, y_train, **config.dataset.get('kwargs', dict()))
      #train_dataset = datasets.OversamplingDataset(x_train, y_train)
      #train_dataset = datasets.SinchDataset(x_train, y_train)
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      val_dataset = SinchDataset(x_val, y_val)

      # Ensambling
      n_ensambles = config.general['n_ensambles']
      for ensamble_i in range(n_ensambles):
        iteration.append([])
        train_loss.append([])
        val_loss.append([])
        val_f1.append([])
        ensamble_preds = np.empty((n_ensambles, len(y_val)))
        # Initialize Model
        model = getattr(models, config.model['name'])(nbr_dims, nbr_classes, **config.model.get('kwargs', dict())).to(device)
        #model = models.LinearNet(nbr_dims, nbr_classes, layer_size=128, dropout_rate=0.3)

        # Criterion
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        learning_rate = 10**config.optimizer['log_learning_rate']
        decay=0
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
        #scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.general['patience'])
        
        iter_i = 0
        with trange(n_epochs) as pbar:
            for epoch in pbar:
                #if done:
                #  break
                pbar.set_description(f"Boot {boot_i} - Epoch {epoch}")

def train_step(x, y, M):
    M.optimizer.zero_grad()
    z = M.model(x)
    loss = M.criterion(z, y)
    loss.backward()
    M.optimizer.step()
    return loss

def validate(x, y, M):
    M.model.eval()
    z_pred = M.model(x)
    loss = M.criterion(z_pred, y)
    eval_metrics = dict()
    for name, fun in M.metrics.items():
        eval_metrics[name] = fun(y, z_pred)
    M.model.train()
    return loss, eval_metrics
    
                for x, y in train_dataloader:  
                    train_loss = train_step(x, y, M)
                    iter = iter + 1
                    if i
                    # Reset gradient
                    optimizer.zero_grad()
                    # Backprop
                    z = model(x)
                    loss = criterion(z, y)
                    loss.backward()
                    optimizer.step()
                    
                    iter_i += 1
                    if iter_i % 20 == 0:
                      iteration[-1].append(iter_i)
                      # Save train data 
                      train_loss[-1].append(loss.data)
                      
                      # Validate
                      model.eval()
                      z_pred = model(val_dataset.x)
                      loss = criterion(z_pred, val_dataset.y)
                      eval_score = score(target=val_dataset.y, preds=z_pred,
                                            average='macro', num_classes=nbr_classes).item()
                      model.train()
                      
                      # Save validation data
                      val_loss[-1].append(loss.data)
                      val_f1[-1].append(eval_score)
                      
                      # Scheduler
                      #scheduler.step(eval_score)
                      #if optimizer.state_dict()['param_groups'][0]['lr'] < 1e-5:
                      #  done = True
                      #  break
                      
                    
        # Evaluation
        model.eval()
        z_pred = model(val_dataset.x).detach()
        y_pred = torch.argmax(z_pred, dim=1)

        ensamble_preds[ensamble_i,:] = y_pred.numpy()

