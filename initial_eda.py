import pandas as pd

# Running tallies, filled in chunk by chunk
counts = pd.Series(dtype="int64")
rating_totals = pd.Series(dtype="float64")
rating_counts = pd.Series(dtype="int64")
length_totals = pd.Series(dtype="int64")
length_counts = pd.Series(dtype="int64")
star_counts = pd.Series(
    dtype="int64",
    index=pd.MultiIndex.from_tuples([], names=["vine", "star_rating"])
)
product_vine_counts = pd.Series(
    dtype="int64",
    index=pd.MultiIndex.from_tuples([], names=["product_id", "vine"])
)

for chunk in pd.read_csv(
    "amazon_reviews_us_Beauty_v1_00.tsv",
    sep="\t",
    usecols=["product_id", "vine", "star_rating", "review_body"],
    dtype="string",
    chunksize=100000
):
    chunk["star_rating"] = pd.to_numeric(chunk["star_rating"], errors="coerce")

    # Total Vine vs. non-Vine counts (including any bad values)
    counts = counts.add(chunk["vine"].value_counts(dropna=False), fill_value=0)

    # Keep only valid Vine labels
    valid = chunk[chunk["vine"].isin(["Y", "N"])].copy()

    # Reviews per product per group
    product_vine_counts = product_vine_counts.add(
        valid.groupby(["product_id", "vine"]).size(), fill_value=0
    )

    # Star rating totals for averages
    rating_totals = rating_totals.add(
        valid.groupby("vine")["star_rating"].sum(), fill_value=0
    )
    rating_counts = rating_counts.add(
        valid.groupby("vine")["star_rating"].count(), fill_value=0
    )

    # Review length totals for averages
    lengths = valid["review_body"].fillna("").str.len()
    length_totals = length_totals.add(
        lengths.groupby(valid["vine"]).sum(), fill_value=0
    )
    length_counts = length_counts.add(
        lengths.groupby(valid["vine"]).count(), fill_value=0
    )

    # Star rating distribution
    star_counts = star_counts.add(
        valid.groupby(["vine", "star_rating"]).size(), fill_value=0
    )

# ---- Results ----
print("Review counts:")
print(counts.astype(int))

print("\nAverage star ratings:")
print(rating_totals / rating_counts)

print("\nAverage review length (characters):")
print(length_totals / length_counts)

print("\nStar rating percentages:")
star_pct = star_counts / star_counts.groupby(level="vine").sum() * 100
print(star_pct.unstack().round(3))

# Products reviewed by both groups
groups_per_product = product_vine_counts.groupby(level="product_id").size()
both = groups_per_product[groups_per_product == 2]
print("\nProducts with both Vine and non-Vine reviews:", len(both))

matched = product_vine_counts[
    product_vine_counts.index.get_level_values("product_id").isin(both.index)
].groupby(level="vine").sum()
print("\nReview counts for overlapping products:")
print(matched.astype(int))
