import matplotlib.pyplot as plt

# Xác định các điểm (tạo một hình hộp)
x = [1, 5, 5, 1, 1]
y = [1, 1, 4, 4, 1]

# Vẽ hình chữ nhật
plt.plot(x, y, label="Bounding Box")
plt.fill(x, y, 'b', alpha=0.1)  # Điền màu cho hộp
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Bounding Box Example")

# Chèn các điểm (x0, y0), (x1, y1), ...
plt.scatter([1, 5, 5, 1], [1, 1, 4, 4], color='red')

plt.legend()
plt.grid(True)
plt.savefig('image/bounding_box.png')

