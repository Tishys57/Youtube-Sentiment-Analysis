// Function to handle collapsible behavior
document.addEventListener('DOMContentLoaded', function() {
  var coll = document.getElementsByClassName('collapsible');
  var i;

  for (i = 0; i < coll.length; i++) {
    coll[i].addEventListener('click', function() {
      this.classList.toggle('active');
      var content = this.nextElementSibling;
      if (content.style.display === 'block') {
        content.style.display = 'none';
      } else {
        content.style.display = 'block';
      }
    });
  }
});

// Other JavaScript code for fetching data and displaying results goes here

document.getElementById('submit-btn').addEventListener('click', function() {
  var videoUrl = document.getElementById('comment-url').value.trim();
  if (videoUrl === '') {
    alert('Please enter a YouTube video URL.');
    return;
  }

  fetch('/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ video_url: videoUrl })
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      alert(data.error);
    } else {
      displaySentimentPercentage(data.sentiment_percentage);
      displayComments(data.comments);
      showCollapsibleLists(); // Show collapsible lists after comments are displayed
    }
  })
  .catch(error => {
    console.error('Error:', error);
    alert('An error occurred during analysis.');
  });
});

document.getElementById('download-btn').addEventListener('click', function() {
  window.location.href = '/download';
});

function displaySentimentPercentage(sentimentPercentage) {
  if (typeof sentimentPercentage === 'object' && sentimentPercentage !== null) {
    var positivePercentage = sentimentPercentage.positive_percentage.toFixed(2);
    var neutralPercentage = sentimentPercentage.neutral_percentage.toFixed(2);
    var negativePercentage = sentimentPercentage.negative_percentage.toFixed(2);
    
    document.getElementById('sentiment-percentage').textContent = `Positive: ${positivePercentage}%, Neutral: ${neutralPercentage}%, Negative: ${negativePercentage}%`;
  } else {
    console.error('Invalid sentiment percentage data:', sentimentPercentage);
    document.getElementById('sentiment-percentage').textContent = 'Error: Invalid data';
  }
}

function displayComments(comments) {
  var maxCommentsToShow = 15;

  var positiveComments = comments.filter(comment => comment[1] === 'positive').slice(0, maxCommentsToShow);
  var neutralComments = comments.filter(comment => comment[1] === 'neutral').slice(0, maxCommentsToShow);
  var negativeComments = comments.filter(comment => comment[1] === 'negative').slice(0, maxCommentsToShow);

  updateCommentList('positive-comments', positiveComments);
  updateCommentList('neutral-comments', neutralComments);
  updateCommentList('negative-comments', negativeComments);
}

function updateCommentList(listId, comments) {
  var listContainer = document.getElementById(listId).querySelector('.content');
  listContainer.innerHTML = '';

  if (comments.length === 0) {
    listContainer.innerHTML = '<p>No comments available.</p>';
    return;
  }

  comments.forEach(function(comment) {
    var commentElement = document.createElement('p');
    commentElement.textContent = comment[0];
    listContainer.appendChild(commentElement);
  });
}

function showCollapsibleLists() {
  var collapsibleLists = document.querySelectorAll('.collapsible-list');
  collapsibleLists.forEach(function(list) {
    list.style.display = 'block'; // Show each collapsible list
  });
}