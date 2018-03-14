const chai = require('chai');
const expect = chai.expect;

const fetchMock = require('fetch-mock');
const load = require("../public/index.cjs.js").load;

const mockData = { 'test': 42 }
fetchMock.mock('404', 404);
fetchMock.mock('test.json', JSON.stringify(mockData));
fetchMock.mock('test.csv', `NAME,VALUE,COLOR,DATE
Minna,12,blue,Sep. 25 2009
Hazel,27,teal,Sep. 30 2009`);
fetchMock.mock('test.tsv', `NAME	VALUE	COLOR	DATE
Minna	12	blue	Sep. 25, 2009
Hazel	27	teal	Sep. 30, 2009`);

describe('load()', function () {
  
  it('should throw an error if we get a 404', function (done) {
    load('404')
      .then(result => done(new Error('Completed when it should not have.')))
      .catch(error => done());
  });

  it('should load and parse a json file when the url has a .json extension', function (done) {
    load('test.json').then(result => {
      expect(result).to.have.property('test').and.to.equal(42);
      done();
    });
  });

  it('should load an array of files with the proper file extensions', function (done) {
    load(['test.json', 'test.tsv']).then(results => {
      expect(results).to.have.length(2);
      const json = results[0];
      expect(json).to.have.property('test').and.to.equal(42);
      const tsv = results[1];
      expect(tsv[0]).to.have.property('NAME').and.to.equal('Minna');
      done();
    });
  });

  it('should interrupt a previous call when given the same namespaces', function (done) {
    load('test.json', 'namespace').then(result => {
      done(new Error('Did not cancel'));
    });
    load('test.csv', 'namespace').then(result => {
      expect(result[0]).to.have.property('NAME').and.to.equal('Minna');
      done();
    });
  });

  it('should NOT interrupt a previous call when given two different namespaces', function (done) {
    Promise.all([
      load('test.json', 'namespace1'),
      load('test.csv', 'namespace2')
    ]).then(values => {
      expect(values[0]).to.have.property('test').and.to.equal(42);
      expect(values[1][0]).to.have.property('NAME').and.to.equal('Minna');
      done();
    });
  });

});