<template>
  <div id="app">
    <h1>HITL Animal Classification Interface</h1>
    <div class="image-container">
      <div
        v-for="(image, index) in images"
        :key="index"
        @click="userSelectsCorrectImage(index)"
        :class="{ clickable: guessedIndex !== null }"
      >
        <img :src="image.url" :class="{ guessed: guessedIndex === index }" />
      </div>
    </div>
    <div class="categories">
      <button
        v-for="(category, index) in categories"
        :key="index"
        class="category"
        :class="{ selected: selectedCategoryIndex === index }"
        @click="selectCategory(index)"
      >
        {{ category }}
      </button>
    </div>
    <div v-if="showUserPrompt" class="user-prompt">
      Please select the image you had in mind.
    </div>
    <div class="actions">
      <button class="save-model" @click="saveModelWeights">
        Save Model Weights
      </button>
      <button class="reset-model" @click="resetModel">Reset Model</button>
      <button class="use-trained-model" @click="useTrainedModel">
        Use Trained Model
      </button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      showUserPrompt: false,
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
      allowUserSelection: false,
      availableImages: [],
      guessedCertainty: 0,
      selectedCategoryIndex: null,
    };
  },
  mounted() {
    this.loadZooDataset();
  },
  methods: {
    resetModel() {
      fetch("http://127.0.0.1:5001/reset_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.json();
        })
        .then((data) => {
          console.log("Received response from /reset_model:", data);
          alert(data.message);
          this.iterationCount = 0;
          this.loadNewImages();
        })
        .catch((error) => {
          console.error("Error during /reset_model request:", error);
        });
    },
    useTrainedModel() {
      fetch("http://127.0.0.1:5001/use_trained_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.json();
        })
        .then((data) => {
          console.log("Received response from /use_trained_model:", data);
          alert(data.message);
          this.iterationCount = 0;
          this.loadNewImages();
        })
        .catch((error) => {
          console.error("Error during /use_trained_model request:", error);
        });
    },
    userSelectsCorrectImage(index) {
      if (this.showUserPrompt) {
        this.showUserPrompt = false;
      }
      if (this.allowUserSelection) {
        console.log("User selected the correct image:", this.images[index]);
        const correctImage = this.images[index];
        const selectedCategory = this.categories[this.selectedCategoryIndex];
        const feedbackData = {
          correct_image: {
            name: correctImage.name,
            attributes: correctImage.attributes,
            category: selectedCategory,
          },
        };
        console.log(
          "Sending feedback to backend:",
          JSON.stringify(feedbackData)
        );
        fetch("http://127.0.0.1:5001/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(feedbackData),
        })
          .then((response) => {
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then((data) => {
            console.log("Received response from /train:", data);
            this.allowUserSelection = false;
            this.loadNewImages();
          })
          .catch((error) => {
            console.error("Error during /train request:", error);
          });
      }
    },
    loadZooDataset() {
      fetch("/zoo_dataset.json")
        .then((response) => {
          if (!response.ok) {
            throw new Error("Failed to fetch zoo dataset");
          }
          return response.json();
        })
        .then((data) => {
          console.log("Received response from /zoo_dataset:", data);
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
    selectCategory(index) {
      this.selectedCategoryIndex = index;
      if (this.iterationCount < this.maxIterations) {
        const selectedImages = this.images.map((image) => ({
          name: image.name,
          attributes: image.attributes,
        }));

        fetch("http://127.0.0.1:5001/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            selected_images: selectedImages,
            selected_category: this.categories[this.selectedCategoryIndex],
          }),
        })
          .then((response) => {
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then((data) => {
            console.log("Received response from /predict:", data);
            this.iterationCount++;
            this.guessedIndex = data.guessed_index;
            this.allowUserSelection = true;
            setTimeout(() => {
              this.showUserPrompt = true;
            }, 1000);
            if (data.weights && data.guessed_index != null) {
              const guessedProbabilities = data.weights;
              this.guessedCertainty = Math.max(
                0,
                Math.min(guessedProbabilities[data.guessed_index] * 100, 100)
              );
            } else {
              console.warn(
                "Probabilities or guessed index not available in response data."
              );
            }
          })
          .catch((error) => {
            console.error("Error during /predict request:", error);
          });
      } else {
        alert("Maximum number of iterations reached.");
      }
    },
    loadNewImages() {
      this.showUserPrompt = false;
      this.selectedCategoryIndex = null;
      const selectedImages = [];
      const selectedCategories = new Set();
      while (selectedImages.length < 3 && this.availableImages.length > 0) {
        const randomIndex = Math.floor(
          Math.random() * this.availableImages.length
        );
        const randomImage = this.availableImages[randomIndex];
        if (!selectedCategories.has(randomImage.type)) {
          selectedImages.push(randomImage);
          selectedCategories.add(randomImage.type);
        }
      }

      this.images = selectedImages;
      this.guessedIndex = null;
      this.images.forEach((image) => {
        console.log(
          "Loaded image:",
          image.name,
          "| Type:",
          this.categories[image.type - 1]
        );
      });
    },
    saveModelWeights() {
      const currentDateTime = new Date().toISOString().replace(/[:.]/g, "-");
      const weightsPath = `/weights/model_weights_${currentDateTime}.pth`;
      fetch(
        `http://127.0.0.1:5001/save_model_weights?weights_path=${weightsPath}`,
        { method: "POST", headers: { "Content-Type": "application/json" } }
      )
        .then((response) => {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.json();
        })
        .then((data) => {
          console.log("Received response from /save_model_weights:", data);
          alert(data.message);
          this.iterationCount = 0;
          this.loadNewImages();
        })
        .catch((error) => {
          console.error("Error during /save_model_weights request:", error);
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
.category.selected {
  background-color: #ffcc00;
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
.reset-model {
  background-color: #ffffff;
  border: 1px solid #000;
  padding: 10px 20px;
  cursor: pointer;
  transition: background-color 0.3s ease;
  margin-left: 10px;
}
.reset-model:hover {
  background-color: #ddd;
}
.use-trained-model {
  background-color: #ffffff;
  border: 1px solid #000;
  padding: 10px 20px;
  cursor: pointer;
  transition: background-color 0.3s ease;
  margin-left: 10px;
}
.use-trained-model:hover {
  background-color: #ddd;
}
.model-info {
  margin-top: 20px;
  padding: 10px;
  background-color: #ffffff;
  border-radius: 8px;
}
.user-prompt {
  position: absolute;
  top: 10%;
  left: 50%;
  transform: translate(-50%, -50%);
  max-width: 400px;
  padding: 15px;
  background-color: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
  border-radius: 8px;
  text-align: center;
}
</style>
