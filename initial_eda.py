import pandas as pd
import numpy as np

# Initialize series to hold counts and totals
counts = pd.Series(dtype="int64")
rating_totals = pd.Series(dtype="float64")
rating_counts = pd.Series(dtype="int64")
length_totals = pd.Series(dtype="int64")
length_counts = pd.Series(dtype="int64")
star_counts = pd.Series(
    dtype="int64",
    index=pd.MultiIndex.from_tuples([], names=["vine", "star_rating"])
)
length_values = {}
product_vine_counts = pd.Series(
    dtype="int64",
    index=pd.MultiIndex.from_tuples([], names=["product_id", "vine"])
)

for chunk in pd.read_csv(
    "amazon_reviews_us_Beauty_v1_00.tsv",
    sep="\t",
    usecols=["product_id", "vine", "star_rating", "review_body"],
    dtype={
        "product_id": "string",
        "vine": "string",
        "star_rating": "string",
        "review_body": "string",
    },
    chunksize=100000
):
    # Convert star_rating to numeric, coercing errors to NaN
    chunk["star_rating"] = pd.to_numeric(chunk["star_rating"], errors="coerce")
    counts = counts.add(
        chunk["vine"].value_counts(dropna=False),
        fill_value=0
    )

    valid_chunk = chunk[chunk["vine"].isin(["Y", "N"])].copy()
    product_vine_counts = product_vine_counts.add(
        valid_chunk.groupby(["product_id", "vine"], dropna=True).size(),
        fill_value=0,
    )
    rating_totals = rating_totals.add(
        valid_chunk.groupby("vine")["star_rating"].sum(),
        fill_value=0
    )
    rating_counts = rating_counts.add(
        valid_chunk.groupby("vine")["star_rating"].count(),
        fill_value=0
    )

    review_lengths = valid_chunk["review_body"].fillna("").str.len()
    length_totals = length_totals.add(
        review_lengths.groupby(valid_chunk["vine"]).sum(),
        fill_value=0
    )
    length_counts = length_counts.add(
        review_lengths.groupby(valid_chunk["vine"]).count(),
        fill_value=0
    )

    length_data = valid_chunk.assign(review_length=review_lengths)
    for group, values in length_data.groupby(
        ["star_rating", "vine"]
    )["review_length"]:
        length_values.setdefault(group, []).append(
            values.to_numpy(dtype="int32")
        )

    chunk_star_counts = valid_chunk.groupby(["vine", "star_rating"]).size()
    star_counts = star_counts.add(chunk_star_counts, fill_value=0)

print(counts.astype(int))

# Compare average ratings
print("Average star ratings:")
print(rating_totals / rating_counts)

# Compare review lengths
print("\nAverage review length (characters):")
print(length_totals / length_counts)

# Compare star rating distributions
print("\nStar rating percentages:")
star_percentages = star_counts / star_counts.groupby(level="vine").sum() * 100
print(star_percentages.unstack().round(3))


# Compare review length by Vine status and star rating
length_comparison = pd.DataFrame.from_dict(
    {
        group: {
            "count": len(values),
            "mean": values.mean(),
            "median": np.median(values),
        }
        for group, arrays in length_values.items()
        for values in [np.concatenate(arrays)]
    },
    orient="index",
).rename_axis(["star_rating", "vine"])

print(length_comparison)


# Count products reviewed by both groups
product_counts = product_vine_counts.groupby(level="product_id").size()

both_groups = product_counts[product_counts == 2]

print("Products with both Vine and non-Vine reviews:", len(both_groups))

print("\nReview counts for overlapping products:")
matched_review_counts = product_vine_counts[
    product_vine_counts.index.get_level_values("product_id").isin(both_groups.index)
].groupby(level="vine").sum()
print(matched_review_counts.astype(int))
