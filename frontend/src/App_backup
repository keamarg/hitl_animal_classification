<template>
  <div id="app">
    <h1>HITL Animal Classification Interface</h1>
    <div class="image-container">
      <div v-for="(image, index) in images" :key="index">
        <img :src="image.url" :class="{ guessed: guessedIndex === index }" />
      </div>
    </div>
    <div class="categories">
      <button
        v-for="(category, index) in categories"
        :key="index"
        class="category"
        @click="selectCategory(index)"
      >
        {{ category }}
      </button>
    </div>
    <div class="actions">
      <button class="save-model" @click="saveModelWeights">
        Save Model Weights and Start Fresh
      </button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      images: [],
      categories: [
        "Mammal",
        "Bird",
        "Reptile",
        "Fish",
        "Amphibian",
        "Insect",
        "Invertebrate",
      ],
      iterationCount: 0,
      maxIterations: 50,
      guessedIndex: null,
      availableImages: [],
    };
  },
  mounted() {
    this.loadZooDataset();
  },
  methods: {
    loadZooDataset() {
      console.log("Loading zoo dataset...");
      fetch("/zoo_dataset.json")
        .then((response) => {
          console.log(
            "Fetched zoo_dataset.json:",
            response.status,
            response.statusText
          );
          if (!response.ok) {
            throw new Error("Failed to fetch zoo dataset");
          }
          return response.json();
        })
        .then((data) => {
          console.log("Zoo dataset loaded:", data);
          this.availableImages = data.map((item) => {
            return {
              name: item.animal_name,
              type: item.type, // Keep track of the animal type to ensure unique categories
              url: `/animal_images/${item.animal_name}.jpg`,
              attributes: Object.values(item).slice(1, -1), // Exclude 'animal_name' (first) and 'animal_type' (last)
            };
          });
          this.loadNewImages();
        })
        .catch((error) => {
          console.error("Error loading zoo dataset:", error);
        });
    },
    selectCategory() {
      if (this.iterationCount < this.maxIterations) {
        const selectedImages = this.images.map((image) => ({
          name: image.name,
          attributes: image.attributes,
        }));

        fetch("http://127.0.0.1:5001/train", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            selected_images: selectedImages,
          }),
        })
          .then((response) => {
            console.log("Received response from /train endpoint:", response);
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then((data) => {
            console.log("Data received from /train endpoint:", data);
            this.iterationCount++;
            this.guessedIndex = data.guessed_index;
            if (data.weights && data.guessed_index != null) {
              const weight = data.weights[data.guessed_index];
              const totalWeight = data.weights.reduce(
                (acc, val) => acc + val,
                0
              );
              const percentageCertainty = (
                (weight / totalWeight) *
                100
              ).toFixed(2);
              console.log(
                `Model guessed category with weight: ${weight} and certainty: ${percentageCertainty}%`
              );
            } else {
              console.warn(
                "Weights or guessed index not available in response data."
              );
            }
            setTimeout(() => {
              this.loadNewImages();
            }, 2000);
          })
          .catch((error) => {
            console.error("Error during /train request:", error);
          });
      } else {
        alert("Maximum number of iterations reached.");
      }
    },
    loadNewImages() {
      console.log("Loading new images...");
      const selectedImages = [];
      const selectedCategories = new Set();
      while (selectedImages.length < 3 && this.availableImages.length > 0) {
        const randomIndex = Math.floor(
          Math.random() * this.availableImages.length
        );
        const randomImage = this.availableImages[randomIndex];
        console.log(
          "Animal Name:",
          randomImage.name,
          "| Animal Type:",
          randomImage.type,
          "| Animal Attributes:",
          randomImage.attributes
        );
        if (!selectedCategories.has(randomImage.type)) {
          selectedImages.push(randomImage);
          selectedCategories.add(randomImage.type);
        }
      }

      this.images = selectedImages;
      console.log("Selected images:", this.images);
      this.guessedIndex = null;
    },
    saveModelWeights() {
      const currentDateTime = new Date().toISOString().replace(/[:.]/g, "-");
      const weightsPath = `/weights/model_weights_${currentDateTime}.pth`;
      fetch(`http://127.0.0.1:5001/save_model?weights_path=${weightsPath}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      })
        .then((response) => {
          console.log("Received response from /save_model endpoint:", response);
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.json();
        })
        .then((data) => {
          console.log("Data received from /save_model endpoint:", data.message);
          alert(data.message);
          this.iterationCount = 0;
          this.loadNewImages();
        })
        .catch((error) => {
          console.error("Error during /save_model request:", error);
        });
    },
  },
};
</script>

<style scoped>
body {
  font-family: Arial, sans-serif;
  padding: 20px;
  background-color: #f0a15f;
}
.image-container {
  display: flex;
  justify-content: space-around;
  margin-bottom: 20px;
  padding: 10px;
  background-color: #f0a15f;
  border-radius: 8px;
}
.image-container img {
  width: 200px;
  height: 200px;
  object-fit: cover;
  border: 5px solid transparent;
  transition: border 0.3s ease;
}
.image-container img.guessed {
  border-color: red;
}
.categories {
  display: flex;
  justify-content: space-around;
  margin-top: 20px;
  padding: 10px;
  background-color: #f0a15f;
  border-radius: 8px;
}
.category {
  background-color: #ffffff;
  border: 1px solid #000;
  padding: 10px 20px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}
.category:hover {
  background-color: #ddd;
}
.actions {
  display: flex;
  justify-content: center;
  margin-top: 20px;
}
.save-model {
  background-color: #ffffff;
  border: 1px solid #000;
  padding: 10px 20px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}
.save-model:hover {
  background-color: #ddd;
}
</style>
