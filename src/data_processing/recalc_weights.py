
import numpy as np

counts = {
    0: 7657, 1: 143, 2: 26, 3: 7, 4: 1446, 5: 15, 6: 0, 7: 4,
    8: 1572, 9: 87, 10: 31, 11: 14, 12: 802, 13: 61, 14: 160, 15: 109
}
total_samples = sum(counts.values())

cw_dict = {}
for i in range(16):
    count = counts[i]
    if count > 0:
        cw_dict[i] = total_samples / (16.0 * count)
    else:
        cw_dict[i] = 1.0 # Fallback used in script

avg_w = sum(cw_dict.values()) / 16.0
norm_cw = {k: v / avg_w for k, v in cw_dict.items()}

print(f"Total Samples: {total_samples}")
print("Class | Count | Percentage | Normalized Weight")
print("-" * 50)
for i in range(16):
    p = (counts[i] / total_samples) * 100
    print(f"{i:5d} | {counts[i]:5d} | {p:9.2f}% | {norm_cw[i]:.4f}")
