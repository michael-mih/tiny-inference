def add_wandb_args(parser):
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument(
        "--wandb-project",
        default=None,
        help="Enable W&B logging under this project.",
    )
    group.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional W&B entity or team.",
    )
    group.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name.",
    )
    group.add_argument(
        "--wandb-group",
        default=None,
        help="Optional W&B run group.",
    )
    group.add_argument(
        "--wandb-tags",
        action="append",
        default=[],
        help="Comma-separated W&B tags. May be supplied multiple times.",
    )
    group.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Optional W&B mode. Use offline to sync later.",
    )
    group.add_argument(
        "--wandb-dir",
        default=None,
        help="Optional local directory for W&B run files.",
    )


def _parse_tags(raw_tags):
    tags = []
    for raw_tag in raw_tags or []:
        tags.extend(tag.strip() for tag in raw_tag.split(",") if tag.strip())
    return tags or None


def _clean_record(record):
    return {key: value for key, value in record.items() if value is not None}


class DisabledWandbLatencyLogger:
    enabled = False

    def log(self, record):
        return None

    def update_summary(self, values):
        return None

    def finish(self):
        return None


class WandbLatencyLogger:
    enabled = True

    def __init__(self, run, wandb_module, table_key):
        self.run = run
        self._wandb = wandb_module
        self._table_key = table_key
        self._records = []
        self._columns = []
        self._step = 0

    def log(self, record):
        normalized_record = {
            "wandb_step": self._step,
            **record,
        }
        self._records.append(normalized_record)
        for key in normalized_record:
            if key not in self._columns:
                self._columns.append(key)

        self.run.log(_clean_record(normalized_record), step=self._step)
        self._step += 1

    def update_summary(self, values):
        for key, value in _clean_record(values).items():
            self.run.summary[key] = value

    def finish(self):
        if self._records:
            rows = [
                [record.get(column) for column in self._columns]
                for record in self._records
            ]
            table = self._wandb.Table(columns=self._columns, data=rows)
            self.run.log({self._table_key: table})
        self.run.finish()


def init_wandb_latency_logger(args, *, config, job_type, table_key):
    if not args.wandb_project:
        return DisabledWandbLatencyLogger()

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging was requested, but the 'wandb' package is not installed. "
            "Install it with `pip install wandb`."
        ) from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "group": args.wandb_group,
        "tags": _parse_tags(args.wandb_tags),
        "mode": args.wandb_mode,
        "dir": args.wandb_dir,
        "job_type": job_type,
        "config": config,
    }
    init_kwargs = _clean_record(init_kwargs)
    run = wandb.init(**init_kwargs)
    return WandbLatencyLogger(run, wandb, table_key)
