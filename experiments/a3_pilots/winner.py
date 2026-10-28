"""
Who won the procurement, from information available before the contract stage.

Pilots 4–5 took the winner from the contracts table, but the contract signer is partly an outcome: in
procurements with bids the first contract's supplier did not apply at all in 27 % of successful and 63 % of
failed procurements (contracts concluded after a refusal or outside the original competition). So:

  * announced supplier (trd_buy.biinSupplier) for single-source purchases — matches the contract signer in
    99.8–100 % of cases and is known at publication;
  * otherwise the lowest bidder by discounted price, per lot, most frequent across lots — matches the
    signer in 87.5 % of successful procurements where the signer applied, and is known at the bid deadline.

The customer comes from the lots (known at publication), not from contracts.
"""
APP_LOTS_SCHEMA = ('[{"lotId":"BIGINT","statusId":"BIGINT","price":"DOUBLE","amount":"DOUBLE",'
                   '"discountValue":"DOUBLE","discountPrice":"DOUBLE"}]')


def create_winner_table(con, raw: str) -> None:
    """Creates table `award(id, winner, cust, winner_source)` for every procurement in trd_buy."""
    con.sql(f"""CREATE OR REPLACE TABLE award AS
    WITH b AS (SELECT id, nullif(biinSupplier, '') AS announced FROM read_parquet('{raw}/trd_buy.parquet')
               QUALIFY row_number() OVER (PARTITION BY id ORDER BY lastUpdateDate DESC) = 1),
    al AS (SELECT a.buyId AS id, j.lotId AS lot_id, a.supplierBinIin AS sup,
                  CASE WHEN j.discountPrice > 0 THEN j.discountPrice ELSE j.price END AS p
           FROM read_parquet('{raw}/trd_app.parquet') a, unnest(from_json(a.AppLots, '{APP_LOTS_SCHEMA}')) t(j)
           WHERE j.price > 0),
    lw AS (SELECT id, lot_id, arg_min(sup, p) AS sup FROM al GROUP BY 1, 2),
    bw AS (SELECT id, mode(sup) AS sup FROM lw GROUP BY 1),
    cu AS (SELECT trdBuyId AS id, arg_min(customerBin, id) AS cust FROM read_parquet('{raw}/lots.parquet') GROUP BY 1)
    SELECT b.id, coalesce(b.announced, bw.sup) AS winner, cu.cust,
           CASE WHEN b.announced IS NOT NULL THEN 'announced' WHEN bw.sup IS NOT NULL THEN 'lowest_bid' END AS winner_source
    FROM b LEFT JOIN bw USING (id) LEFT JOIN cu USING (id)""")
