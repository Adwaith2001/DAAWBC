class FixedBidAgent:
    """
    Always bids the same fixed amount.
    Good for testing that the environment runs end-to-end.
    """
    def __init__(self, bid: float):
        self.bid = float(bid)

    def act(self, state, impression=None) -> float:
        # state is a tensor, impression is a dict with pctr & market_price (optional here)
        return self.bid


class LinearPctrAgent:
    """
    Bid proportional to pCTR, capped at some maximum.
    bid = min(k * pCTR, cap)
    """
    def __init__(self, k: float, cap: float):
        self.k = float(k)
        self.cap = float(cap)

    def act(self, state, impression: dict | None) -> float:
        if impression is None or "pctr" not in impression:
            return 0.0
        pctr = float(impression["pctr"])
        return min(self.k * pctr, self.cap)
