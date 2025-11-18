self.writer = None
if "tensorboard" in opt and opt["tensorboard"]:
    tb_dir = os.path.join(log_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    self.writer = SummaryWriter(tb_dir)
