from __future__ import annotations


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(prog="cointrainer")
    p.add_argument("--version", action="version", version="cointrainer CLI")

    sub = p.add_subparsers(dest="cmd", required=True)

    # import-csv7
    imp = sub.add_parser("import-csv7", help="Ingest headerless 7-col CSV (ts, o, h, l, c, v, trades)")
    imp.add_argument("--file", required=True, help="Path to the source CSV")
    imp.add_argument("--symbol", default="XRPUSD")
    imp.add_argument("--out", default=None, help="Output prefix (e.g., data\\XRPUSD_1m)")

    # csv-train
    tr = sub.add_parser("csv-train", help="Train a regime model directly from CSV7 or normalized CSV")
    tr.add_argument("--file", required=True, help="CSV7 (7-col) or normalized CSV with OHLCV(+trades)")
    tr.add_argument("--symbol", default="XRPUSD")
    tr.add_argument("--horizon", type=int, default=15)
    tr.add_argument("--hold", type=float, default=0.0015)
    tr.add_argument("--publish", action="store_true", help="Publish to registry if configured")

    args = p.parse_args()

    if args.cmd == "import-csv7":
        from cointrainer.io.csv7 import read_csv7
        df = read_csv7(args.file)
        prefix = Path(args.out) if args.out else Path(f"{args.symbol}_1m")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        norm_csv = prefix.with_suffix(".normalized.csv")
        df.to_csv(norm_csv, index=True)
        print(
            f"Wrote normalized CSV: {norm_csv}  rows={len(df):,}  range={df.index[0]} .. {df.index[-1]}"
        )
        try:
            df.to_parquet(prefix.with_suffix(".parquet"))
            print(f"Wrote Parquet:       {prefix.with_suffix('.parquet')}")
        except Exception as e:  # pragma: no cover - best effort for optional dependency
            print(f"Parquet not written (install pyarrow or fastparquet): {e}")
        return

    if args.cmd == "csv-train":
        from cointrainer.train.local_csv import TrainConfig, train_from_csv7

        cfg = TrainConfig(
            symbol=args.symbol,
            horizon=args.horizon,
            hold=args.hold,
            publish_to_registry=args.publish,
        )
        # Detect if it's CSV7 (7 columns, no header) vs normalized (has header 'open',...)
        # We try CSV7 reader first; if it fails due to header mismatch, assume normalized CSV.
        try:
            _ = train_from_csv7(args.file, cfg)
            print("Training completed from CSV7 source.")
        except Exception:
            # second chance: try normalized CSV path
            import pandas as pd
            from cointrainer.train.local_csv import (
                FEATURE_LIST,
                _fit_model,
                _maybe_publish_registry,
                _save_local,
                make_features,
                make_labels,
            )

            df = pd.read_csv(args.file, parse_dates=[0], index_col=0)
            df.index.name = "ts"
            df = df.sort_index()
            X = make_features(df).dropna()
            y = make_labels(df.loc[X.index, "close"], cfg.horizon, cfg.hold).dropna()
            m = y.index.intersection(X.index)
            X = X.loc[m]
            y = y.loc[m]
            model = _fit_model(X, y)
            meta = {
                "schema_version": "1",
                "feature_list": FEATURE_LIST,
                "label_order": [-1, 0, 1],
                "horizon": f"{cfg.horizon}m",
                "thresholds": {"hold": cfg.hold},
                "symbol": cfg.symbol,
            }
            path = _save_local(model, cfg, meta)
            try:
                import io, joblib

                buf = io.BytesIO()
                joblib.dump(model, buf)
                _maybe_publish_registry(buf.getvalue(), meta, cfg)
            except Exception:
                pass
            print(f"Training completed from normalized CSV. Model: {path}")
        return


if __name__ == "__main__":
    main()
