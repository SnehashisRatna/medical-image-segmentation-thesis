from src.data.metadata_scanner import MetadataScanner


def main() -> None:

    scanner = MetadataScanner(
        "data/raw/CHAOS_Train_Sets/Train_Sets"
    )

    scanner.scan()

    df = scanner.to_dataframe()

    print("\n" + "=" * 70)
    print("CHAOS DATASET SUMMARY")
    print("=" * 70)

    print(df)

    print("\n")
    print(df.info())

    scanner.save_csv(
        "outputs/reports/dataset_summary.csv"
    )


if __name__ == "__main__":
    main()