RECS = {
    "Premium Customers": ["VIP membership", "Early product access", "Exclusive offers"],
    "Loyal Customers": ["Loyalty points", "Referral rewards", "Cross-sell bundles"],
    "Regular Customers": ["Seasonal promotions", "Upsell nudges"],
    "At-Risk Customers": ["Re-engagement emails", "Personalized discounts", "Win-back survey"],
}

def get_recommendations(segment):
    return RECS.get(segment, ["Monitor behaviour"])