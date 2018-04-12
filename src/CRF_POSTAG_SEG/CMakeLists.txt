
include_directories(
  model
)

add_executable(POSTAG_SEG NNCRFSegmentor.cpp)
target_link_libraries(POSTAG_SEG ${LIBS})

