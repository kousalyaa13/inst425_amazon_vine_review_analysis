
import pandas as pd


product_vine_counts = pd.Series(
    dtype="int64",
    index=pd.MultiIndex.from_tuples([], names=["product_id", "vine"]),
)

for chunk in pd.read_csv(
    "amazon_reviews_us_Beauty_v1_00.tsv",
    sep="\t",
    usecols=["product_id", "vine"],
    dtype={"product_id": "string", "vine": "string"},
    chunksize=100000,
):
    valid_chunk = chunk[chunk["vine"].isin(["Y", "N"])]
    product_vine_counts = product_vine_counts.add(
        valid_chunk.groupby(["product_id", "vine"], dropna=True).size(),
        fill_value=0,
    )


# Count Vine and non-Vine reviews per product
product_review_counts = product_vine_counts.unstack(fill_value=0)
for vine_status in ["Y", "N"]:
    if vine_status not in product_review_counts:
        product_review_counts[vine_status] = 0

# Products with at least 5 reviews from each group
eligible_products = product_review_counts[
    (product_review_counts["Y"] >= 5) &
    (product_review_counts["N"] >= 5)
]

print("Products with at least 5 reviews in each group:")
print(len(eligible_products))

print("\nTotal reviews for those products:")
print(product_review_counts.loc[eligible_products.index, ["Y", "N"]].sum().astype(int))
